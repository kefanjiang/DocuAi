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
import { traceable, getCurrentRunTree } from "langsmith/traceable";

const QUESTION = "Update an existing pet";
const FOLLOW_UP_QUESTION = "Tell me more about the request";
const ORG_ID = "tappu_8ede7240-110c-4dad-a417-d5403b5c94f1";

function buildVectorRetriever(
  orgId: string,
  redisVectorService: RedisVectorService
) {
  // return redisVectorService.asRetriever(3, [orgId]);
  return redisVectorService.asRetriever(7);
}

const goldLabels = {
  "Update an existing pet": ["updatePet", "updatePetWithForm"],
};

// @ts-ignore
(async () => {
  const orgId = StringUtils.removeHyphensAndUnderscores(ORG_ID);
  const redisClient = await RedisClient.create();
  const historyService = new HistoryService(redisClient);
  const bufferMemory = new BufferMemory({
    memoryKey: "chat_history",
    returnMessages: true,
    chatHistory: historyService.getRedisChatMessageHistoryById(orgId),
  });

  const openAIService = new OpenAIService();
  const redisVectorService = new RedisVectorService(openAIService, redisClient);

  const retriever = buildVectorRetriever(orgId, redisVectorService);

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

  const chatHistory = await bufferMemory.chatHistory.getMessages();
  const tracedRetriever = traceable(
    async () => {
      const docs = await historyAwareRetriever.invoke({
        input: QUESTION,
        chat_history: chatHistory,
      });

      const relevantDocs = docs.filter((doc) => {
        const operationId = doc.metadata?.operationId;
        return (
          operationId &&
          (goldLabels[QUESTION]?.includes(operationId) ||
            goldLabels[FOLLOW_UP_QUESTION]?.includes(operationId))
        );
      });

      const precision = relevantDocs.length / docs.length;
      const recall = relevantDocs.length / goldLabels[QUESTION].length;

      const runTree = getCurrentRunTree();
      runTree.extra.metadata = {
        ...runTree.extra.metadata,
        precision: precision.toFixed(2),
        recall: recall.toFixed(2),
      };
    },
    {
      name: "History Aware Retriever",
      tags: ["retriever"],
    }
  );
  await tracedRetriever();

  // const qaPrompt = ChatPromptTemplate.fromMessages([
  //   SystemMessagePromptTemplate.fromTemplate(DEFAULT_QA_PROMPTS),
  //   new MessagesPlaceholder("chat_history"),
  //   HumanMessagePromptTemplate.fromTemplate("{input}"),
  // ]);

  // const combineDocsChain = await createStuffDocumentsChain({
  //   llm: openAIService.createChatOpenAI(),
  //   prompt: qaPrompt,
  // });

  // const retrievalChain = await createRetrievalChain({
  //   retriever: historyAwareRetriever,
  //   combineDocsChain: combineDocsChain,
  // });

  // const chatHistory = await bufferMemory.chatHistory.getMessages();
  // const result = await retrievalChain.invoke({
  //   input: QUESTION,
  //   chat_history: chatHistory,
  // });

  // await bufferMemory.saveContext(
  //   { input: QUESTION },
  //   { output: result.answer }
  // );

  await redisClient.disconnect();
})();
