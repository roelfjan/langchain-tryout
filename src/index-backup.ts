import * as dotenv from "dotenv";
import { OpenAI } from "langchain";
import { PromptTemplate } from "langchain/prompts";
import {
  ConversationalRetrievalQAChain,
  LLMChain,
  RetrievalQAChain,
} from "langchain/chains";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { SerpAPI } from "langchain/tools";
import { Calculator } from "langchain/tools/calculator";
import { HNSWLib } from "langchain/vectorstores/hnswlib";

import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

dotenv.config();

// const model = new OpenAI({
//   modelName: "gpt-3.5-turbo",
//   openAIApiKey: process.env.OPENAI_API_KEY,
// });

export const run = async () => {
  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({ modelName: "gpt-4" });
  /* Load in the file we want to do question answering over */
  // const text = fs.readFileSync("hymn.txt", "utf8");
  // /* Split the text into chunks */
  // const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  // const docs = await textSplitter.createDocuments([text]);

  const folderPath = "./data/";
  const files = fs.readdirSync(folderPath);

  const texts: string[] = [];

  files.forEach((file) => {
    const filePath = folderPath + file;

    if (filePath.endsWith(".txt")) {
      const text = fs.readFileSync(filePath, "utf8");
      texts.push(text);
    }
  });

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments(texts);

  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  /* Create the chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever()
  );
  /* Ask it a question */
  const question =
    "Wat heb je nodig om de ontwikkeling van een app succesvol te maken?";
  const res = await chain.call({ question, chat_history: [] });
  console.log(res);
  /* Ask it a follow up question */
  // const chatHistory = question + res.text;
  // const followUpRes = await chain.call({
  //   question: "Was that nice?",
  //   chat_history: chatHistory,
  // });
  // console.log(followUpRes);

  // const model = new OpenAI({ temperature: 0.9 });

  // const text = fs.readFileSync("hymn.txt", "utf8");
  // const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  // const docs = await textSplitter.createDocuments([text]);

  // // Create a vector store from the documents.
  // const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  // // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
  // const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  // const res = await chain.call({
  //   query: "Wat is het telefoonnummer van Chris?",
  // });
  // console.log({ res });

  // const tools = [
  //   new SerpAPI(process.env.SERPAPI_API_KEY, {
  //     location: "Austin,Texas,United States",
  //     hl: "en",
  //     gl: "us",
  //   }),
  //   new Calculator(),
  // ];

  // const executor = await initializeAgentExecutorWithOptions(tools, model, {
  //   agentType: "zero-shot-react-description",
  // });
  // console.log("Loaded agent.");

  // const input = "Wanneer werd Napoli voor het laatst kampioen van Italië?";
  // console.log(`Executing with input "${input}"...`);

  // const result = await executor.call({ input });

  // console.log(`Got output ${result.output}`);

  // const template = "What is a good name for a company that makes {product}?";
  // const prompt = new PromptTemplate({
  //   template,
  //   inputVariables: ["product"],
  // });

  // const chain = new LLMChain({ llm: model, prompt });

  // const res = await chain.call({ product: "colorful socks" });

  // console.log(res.text);
};

run();
