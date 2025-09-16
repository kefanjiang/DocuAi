import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { OpenAIService } from "./langchain/openai/openai.service";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import { DEFAULT_QA_PROMPTS } from "./langchain/prompts/qa.prompt";
import { RedisClient } from "./redis/redis.client";
import { RedisVectorService } from "./langchain/vectorstores/redis.vectorstore";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { BufferMemory } from "langchain/memory";
import { HistoryService } from "./langchain/history/history.service";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { DEFAULT_REPHRASE_PROMPTS } from "./langchain/prompts/rephrase.prompt";
import { StringUtils } from "./utils/string";
import { FilterChatHistoryService } from "./langchain/history/filter.chat.history";
import { BaseMessage } from "@langchain/core/messages";

const QUESTION = "Update an existing pet";
const UNRELATED_QUESTION = "How is the weather today?";
const FOLLOW_UP_QUESTION = "Tell me more about the request";
const ORG_ID = "tappu_8ede7240-110c-4dad-a417-d5403b5c94f1";

function buildVectorRetriever(
  orgId: string,
  redisVectorService: RedisVectorService
) {
  return redisVectorService.asRetriever(3, [orgId]);
  //return redisVectorService.asRetriever(3);
}

// @ts-ignore
(async () => {
  const redisClient = await RedisClient.create();
  const historyService = new HistoryService(redisClient);
  const bufferMemory = new BufferMemory({
    memoryKey: "chat_history",
    returnMessages: true,
    chatHistory: historyService.getRedisChatMessageHistoryById(ORG_ID),
  });

  const openAIService = new OpenAIService();
  const redisVectorService = new RedisVectorService(openAIService, redisClient);

  const retriever = buildVectorRetriever(ORG_ID, redisVectorService);

  const rephrasePrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(DEFAULT_REPHRASE_PROMPTS),
    new MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    retriever: retriever,
    llm: openAIService.createChatOpenAI(),
    rephrasePrompt: rephrasePrompt,
  });

  const qaPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(DEFAULT_QA_PROMPTS),
    new MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const combineDocsChain = await createStuffDocumentsChain({
    llm: openAIService.createChatOpenAI(),
    prompt: qaPrompt,
  });

  const retrievalChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: combineDocsChain,
  });

  const rawChatHistory = await bufferMemory.chatHistory.getMessages();
  console.log(`📝 Raw chat history length: ${rawChatHistory.length}`);
  console.log("📝 Raw chat history sample:", rawChatHistory.slice(0, 4));

  //filter AI no answer history
  const chatHistory: BaseMessage[] =
    FilterChatHistoryService.filter(rawChatHistory);
  console.log(`✅ Filtered chat history length: ${chatHistory.length}`);
  console.log("✅ Filtered chat history sample:", chatHistory.slice(0, 4));

  const result = await retrievalChain.invoke({
    input: FOLLOW_UP_QUESTION,
    chat_history: chatHistory,
  });

  await bufferMemory.saveContext(
    { input: FOLLOW_UP_QUESTION },
    {
      output: result.answer,
    }
  );
  console.log(result.answer);

  await redisClient.disconnect();
})();
