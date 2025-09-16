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
  QA_PROMPTS_WITH_IDENTITY_INSTRUCTIONS,
  SIMPLE_QA_PROMPTS,
} from "./langchain/prompts/qa.prompt";
import { RedisClient } from "./redis/redis.client";
import { RedisVectorService } from "./langchain/vectorstores/redis.vectorstore";

//const QUESTION = "Update an existing pet";
//const QUESTION = "Assign the user a role";
//const QUESTION = "Pet store";
const QUESTION = "Tell me about the pet store API";
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
  const retrievedDocs = await retriever.invoke(QUESTION);
  retrievedDocs.forEach((item) => {
    console.log(item.metadata);
  });

  const qaPrompt = ChatPromptTemplate.fromMessages([
    //SystemMessagePromptTemplate.fromTemplate(SIMPLE_QA_PROMPTS),
    SystemMessagePromptTemplate.fromTemplate(
      QA_PROMPTS_WITH_IDENTITY_INSTRUCTIONS
    ),
    //SystemMessagePromptTemplate.fromTemplate(QA_PROMPTS_WITH_FEW_SHOTS),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const combineDocsChain = await createStuffDocumentsChain({
    llm: openAIService.createChatOpenAI(),
    prompt: qaPrompt,
  });

  // Mock combineDocsChain.invoke messages
  const messages = await qaPrompt.formatMessages({
    context: retrievedDocs,
    input: QUESTION,
  });
  console.log(messages);

  const result = await combineDocsChain.invoke({
    context: retrievedDocs,
    input: QUESTION,
  });
  console.log(result);
  await redisClient.disconnect();
})();
