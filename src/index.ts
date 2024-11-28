import { z, genkit,indexerRef, run } from 'genkit';
import { Document } from 'genkit/retriever';
import { googleAI, gemini15Flash,textEmbeddingGecko001 } from '@genkit-ai/googleai'
import pdf from 'pdf-parse'
import fs from 'fs'
import { devLocalIndexerRef, devLocalVectorstore, devLocalRetrieverRef } from '@genkit-ai/dev-local-vectorstore';
import { chunk } from 'llm-chunk';

import dotenv from 'dotenv'
dotenv.config()



const ai = genkit({
  plugins: [
    googleAI({apiKey: process.env.GOOGLE_GENAI_API_KEY}),
    devLocalVectorstore([
      {
        embedder: textEmbeddingGecko001,
        indexName: "facts",
      },
    ]),
  ],
  model: gemini15Flash,
});

export const PdfIndexer = devLocalIndexerRef('facts');
const chunkingConfig = {
  minLength: 1000,
  maxLength: 2000,
  splitter: 'sentence',
  overlap: 100,
  delimiters: '',
} as any;


const indexMenu = ai.defineFlow(
  {
    name: 'indexMenu',
    inputSchema: z.string()
  },
  async () => {
    

    // Read the pdf.
    const pdfTxt = await pdf(fs.readFileSync("nish-js.pdf"));

    // Divide the pdf text into segments.
    const chunks = await run('chunk-it', async () =>
      chunk(pdfTxt.text, chunkingConfig)
    );

    // Convert chunks of text into documents to store in the index.
    // const documents = chunks.map((text) => {
      // return Document.fromText(text, { filePath });
    // });

    // Add documents to the index.
    await ai.index({
      indexer: PdfIndexer,
      documents: chunks.map((c: string) => Document.fromText(c)),
    });
  }
);

export const retriever = devLocalRetrieverRef('facts')

const helloFlow = ai.defineFlow(
    {name: 'hello', inputSchema: z.string(), outputSchema: z.string()}, 
    async (ques) => {
      const docs = await ai.retrieve({
        retriever: retriever,
        query: ques,
        options: { k:3 }
      })

      const { text } = await ai.generate({
        prompt: ques,
        docs
      });
      return text;
});

// (async () => {
//   console.log(await helloFlow("Hello Gemini"));
// })();
