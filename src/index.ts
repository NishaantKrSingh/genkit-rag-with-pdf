import { z, genkit } from 'genkit';
import { googleAI, gemini15Flash } from '@genkit-ai/googleai'
import pdf from 'pdf-parse'
import fs from 'fs'

const ai = genkit({
  plugins: [googleAI()],
  model: gemini15Flash,
});

const helloFlow = ai.defineFlow(
    {name: 'hello', inputSchema: z.string(), outputSchema: z.string()}, 
    async (prompt) => {
        const pdf_data = await pdf(fs.readFileSync("nish-js.pdf"))

        const { text } = await ai.generate(`${pdf_data.text} \n\n question: ${prompt}`);
        return text;
});

// (async () => {
//   console.log(await helloFlow("Hello Gemini"));
// })();
