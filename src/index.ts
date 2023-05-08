import * as dotenv from "dotenv";
import { OpenAI } from "langchain";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";

import * as fs from "fs";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

dotenv.config();

export const run = async () => {
  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({
    modelName: "gpt-4",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  /* Load in the file we want to do question answering over */
  const folderPath = "./documents/";
  const files = fs.readdirSync(folderPath);

  const texts: string[] = [];

  files.forEach((file) => {
    const filePath = folderPath + file;

    if (filePath.endsWith(".txt")) {
      const text = fs.readFileSync(filePath, "utf8");
      texts.push(text);
    }
  });

  /* Split the text into chunks */
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
  const question = "Welke features zijn er gebouwd voor de HEMA app?";
  const res = await chain.call({ question, chat_history: [] });
  console.log(res);

  /* Ask it a follow up question */
  // const chatHistory = question + res.text;
  // const followUpRes = await chain.call({
  //   question: "Was that nice?",
  //   chat_history: chatHistory,
  // });
  // console.log(followUpRes);
};

run();
