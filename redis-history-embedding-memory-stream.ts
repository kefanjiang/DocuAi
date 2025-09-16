import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { OpenAIService } from './langchain/openai/openai.service';
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { DEFAULT_QA_PROMPTS } from './langchain/prompts/qa.prompt';
import { DEFAULT_REPHRASE_PROMPTS } from './langchain/prompts/rephrase.prompt';
import { DEFAULT_HISTORY_REPHRASE_PROMPTS } from './langchain/prompts/history.rephrase.prompt';
import { RedisClient } from './redis/redis.client';
import { RedisVectorService } from './langchain/vectorstores/redis.vectorstore';
import { HistoryRedisVectorService } from './langchain/vectorstores/redis.history.vectorstore';
import { HistoryService } from './langchain/history/history.service';
import { HistoryEmbeddingService } from './langchain/history/history.embedding';
import { BufferMemory } from 'langchain/memory';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { AIMessage, HumanMessage, BaseMessage } from '@langchain/core/messages';
import { STREAM_EVENT, STREAM_EVENT_NAME } from './constants/stream-event.constant';
import { StreamEventUtilFactory } from './utils/stream-event.util';
import { StringUtils } from './utils/string';
import {ScoreThresholdRetriever} from "langchain/retrievers/score_threshold";

const QUESTION = 'Update an existing pet';
const ORG_ID = 'tappu_8ede7240-110c-4dad-a417-d5403b5c94f1';
const TAG_STUFF_DOCUMENTS_CHAIN = 'stuff_documents_chain';
const STREAM_EVENT_VERSION = 'v1';

function convertDocsToMessages(docs: any[]): BaseMessage[] {
    return docs.map((doc) => {
        const role = doc.metadata?.role || 'user';
        const content = doc.pageContent;
        return role === 'ai' ? new AIMessage(content) : new HumanMessage(content);
    });
}

(async () => {
    const orgId = StringUtils.removeHyphensAndUnderscores(ORG_ID);

    const redisClient = await RedisClient.create();
    const openAIService = new OpenAIService();

    const historyVectorService = new HistoryRedisVectorService(openAIService, redisClient);
    const historyEmbeddingService = new HistoryEmbeddingService(historyVectorService);
    const historyService = new HistoryService(redisClient);
    await historyVectorService.initIndexIfNotExists();
    const bufferMemory = new BufferMemory({
        memoryKey: 'chat_history',
        returnMessages: true,
        chatHistory: historyService.getRedisChatMessageHistoryById(orgId),
    });

    // 1. 用 HistoryAwareRetriever 一步召回并过滤历史上下文（替代原来手动先召回再LLM筛选）
    const historyRephrasePrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(DEFAULT_HISTORY_REPHRASE_PROMPTS),
        new MessagesPlaceholder('chat_history'),
        HumanMessagePromptTemplate.fromTemplate('{input}'),
    ]);

    const historyRetriever = ScoreThresholdRetriever.fromVectorStore(
        historyVectorService.getVectorStore,
        {
            filter: ['*' + orgId + '*'],
            maxK: 10,
            minSimilarityScore: 0.1,
            kIncrement: 3,
        }
    );
    const historyAwareRetriever = await createHistoryAwareRetriever({
        retriever: historyRetriever,
        llm: openAIService.createChatOpenAI(),
        rephrasePrompt: historyRephrasePrompt,
    });

    // 2. 直接用 HistoryAwareRetriever 调用，获得过滤后的历史文档（BaseMessage[]）
    const filteredHistoryDocs = await historyAwareRetriever.invoke({
        input: QUESTION,
        chat_history: [], // 这里如果你有上下文可以传入，否则空数组即可
    });
    const filteredChatHistoryMessages = convertDocsToMessages(filteredHistoryDocs);

    console.log('\n💬 LLM History chat:');
    filteredChatHistoryMessages.forEach((msg, i) => {
        const role = msg._getType() === 'human' ? 'User' : 'Assistant';
        console.log(`  [${i + 1}] ${role}: ${msg.content}`);
    });

    // 3. 构造知识库检索器及重写Prompt
    const knowledgeVectorService = new RedisVectorService(openAIService, redisClient);
    const knowledgeRephrasePrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(DEFAULT_REPHRASE_PROMPTS),
        new MessagesPlaceholder('chat_history'),
        HumanMessagePromptTemplate.fromTemplate('{input}'),
    ]);
    const knowledgeRet = ScoreThresholdRetriever.fromVectorStore(
        knowledgeVectorService.getVectorStore,
        {
            filter: ['*' + orgId + '*'],
            maxK: 3,
            minSimilarityScore: 0.1,
            kIncrement: 3,
        }
    );
    const knowledgeRetriever = await createHistoryAwareRetriever({
        retriever: knowledgeRet,
        llm: openAIService.createChatOpenAI(),
        rephrasePrompt: knowledgeRephrasePrompt,
    });

    // 4. QA 组合链配置
    const qaPrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(DEFAULT_QA_PROMPTS),
        new MessagesPlaceholder('chat_history'),
        HumanMessagePromptTemplate.fromTemplate('{input}'),
    ]);
    const combineDocsChain = await createStuffDocumentsChain({
        llm: openAIService.createChatOpenAI({ tags: [TAG_STUFF_DOCUMENTS_CHAIN] }),
        prompt: qaPrompt,
    });
    const retrievalChain = await createRetrievalChain({
        retriever: knowledgeRetriever,
        combineDocsChain,
    });

    // 5. 流式执行，传入过滤后的历史上下文chat_history
    let result = '';
    const stream = retrievalChain.streamEvents(
        {
            input: QUESTION,
            chat_history: filteredChatHistoryMessages,
        },
        { version: STREAM_EVENT_VERSION }
    );

    const eventHandlers = {
        [STREAM_EVENT.ON_LLM_START]: () => {
            console.log('LLM streaming started');
        },
        [STREAM_EVENT.ON_LLM_STREAM]: (event: any) => {
            if (event.tags.includes(TAG_STUFF_DOCUMENTS_CHAIN)) {
                const partial = StreamEventUtilFactory.getStreamEvent(STREAM_EVENT_VERSION).getLLMStreamAnswer(event);
                if (partial) {
                    result += partial;
                    process.stdout.write(partial);
                }
            }
        },
        [STREAM_EVENT.ON_LLM_END]: async (event: any) => {
            console.log('\nLLM streaming ended');
            if (event.tags.includes(TAG_STUFF_DOCUMENTS_CHAIN)) {
                const finalAnswer = StreamEventUtilFactory.getStreamEvent(STREAM_EVENT_VERSION).getLLMEndAnswer(event);
                if (finalAnswer) {
                    await bufferMemory.saveContext({ input: QUESTION }, { output: finalAnswer });
                    await historyEmbeddingService.embedChatRecord(orgId, QUESTION, finalAnswer);
                    console.log('\nFinal Answer:', finalAnswer);
                    await redisClient.disconnect();
                    console.log('🔌 Redis client disconnected');
                }
            }
        },
        [STREAM_EVENT.ON_CHAIN_STREAM]: (event: any) => {
            if (event.name === STREAM_EVENT_NAME.RETRIEVAL_CHAIN) {
                const partial = StreamEventUtilFactory.getStreamEvent(STREAM_EVENT_VERSION).getChainStreamAnswer(event);
                if (partial) {
                    result += partial;
                    process.stdout.write(partial);
                }
            }
        },
        [STREAM_EVENT.ON_CHAIN_END]: async (event: any) => {
            if (event.name === STREAM_EVENT_NAME.RETRIEVAL_CHAIN) {
                console.log('\nChain streaming ended');
                const finalAnswer = StreamEventUtilFactory.getStreamEvent(STREAM_EVENT_VERSION).getChainEndAnswer(event);
                if (finalAnswer) {
                    await bufferMemory.saveContext({ input: QUESTION }, { output: finalAnswer });
                    await historyEmbeddingService.embedChatRecord(orgId, QUESTION, finalAnswer);
                    console.log('\nFinal Answer:', finalAnswer);
                    await redisClient.disconnect();
                    console.log('🔌 Redis client disconnected');
                }
            }
        },
    };

    for await (const event of stream) {
        const handler = eventHandlers[event.event];
        if (handler) await handler(event);
    }
})();