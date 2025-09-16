import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { OpenAIService } from "./langchain/openai/openai.service";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
import {
  DEFAULT_QA_PROMPTS,
  QA_PROMPTS_WITH_FEW_SHOTS,
} from "./langchain/prompts/qa.prompt";
import { RedisClient } from "./redis/redis.client";
import { RedisVectorService } from "./langchain/vectorstores/redis.vectorstore";
import { createRetrievalChain } from "langchain/chains/retrieval";

const QUESTION = "Update an existing pet";
const ORG_ID = "tappu_8ede7240-110c-4dad-a417-d5403b5c94f1";

function buildVectorRetriever(
  orgId: string,
  redisVectorService: RedisVectorService
) {
  return redisVectorService.asRetriever(3, [orgId]);
}

// @ts-ignore
(async () => {
  const openAIService = new OpenAIService();
  const redisClient = await RedisClient.create();
  const redisVectorService = new RedisVectorService(openAIService, redisClient);

  const retriever = buildVectorRetriever(ORG_ID, redisVectorService);

  const qaPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(QA_PROMPTS_WITH_FEW_SHOTS),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const combineDocsChain = await createStuffDocumentsChain({
    llm: openAIService.createChatOpenAI(),
    prompt: qaPrompt,
  });

  const retrievalChain = await createRetrievalChain({
    retriever: retriever,
    combineDocsChain: combineDocsChain,
  });

  const result = await retrievalChain.invoke({
    input: QUESTION,
  });

  console.log(result);

  await redisClient.disconnect();
})();
