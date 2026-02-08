import { configDotenv } from "dotenv";
configDotenv(); 

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",   
  temperature: 0,
  apiKey: process.env.GOOGLE_API_KEY,
  maxRetries: 2,
});
const addTool = tool(
  async ({ a, b }) => a + b,
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const subtractTool = tool(
  async ({ a, b }) => a - b,
  {
    name: "subtract",
    description: "Subtract b from a",
    schema: z.object({
      a: z.number().describe("Number to subtract from"),
      b: z.number().describe("Number to subtract"),
    }),
  }
);

const multiplyTool = tool(
  async ({ a, b }) => a * b,
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const divideTool = tool(
  async ({ a, b }) => {
    if (b === 0) throw new Error("Division by zero is not allowed");
    return a / b;
  },
  {
    name: "divide",
    description: "Divide a by b (result may be float)",
    schema: z.object({
      a: z.number().describe("Dividend"),
      b: z.number().describe("Divisor (cannot be zero)"),
    }),
  }
);

const tools = [addTool, subtractTool, multiplyTool, divideTool];
const agentExecutor = createReactAgent({
  llm,
  tools,
  messageModifier: `
You are a precise calculator agent.
You MUST use the provided tools for EVERY calculation — never do math mentally or guess.
Break complex expressions into multiple tool calls if needed (e.g., multiply first, then add).
At the very end, after all calculations, respond ONLY with the final numeric result.
No explanation, no words, no units — just the number.

Examples:
Input: What is 15 + 27?
Final output: 42

Input: What is (8 * 3) - 7?
Final output: 17

Input: What is 100 / 4 * 3?
Final output: 75
  `.trim(),
});
async function main() {
  console.log("Gemini Calculator agent started.\n");

  const examples = [
  'hi'
  ];

  for (const question of examples) {
    console.log(`\n┌─ Question: ${question}`);

    try {
      const result = await agentExecutor.invoke({
        messages: [{ role: "user", content: question }],
      });
      let answer = "No final output found";

      if (result.messages && result.messages.length > 0) {
        const lastMsg = result.messages[result.messages.length - 1];
        answer = lastMsg.content?.trim() 
                 || lastMsg.kwargs?.content?.trim() 
                 || lastMsg.text?.trim() 
                 || "Error extracting answer";
      } else if (result.output) {
        answer = result.output.trim();
      }

      console.log("└─ Final answer:", answer);
      console.log("─".repeat(60));
    } catch (err) {
      console.error("Error:", err.message || err);
      if (err?.message?.includes("API key") || err?.message?.includes("quota")) {
        console.error("\n→ Gemini API issue — check your key at https://aistudio.google.com/app/apikey");
      } else if (err?.message?.includes("author") || err?.message?.includes("role")) {
        console.error("→ Possible Gemini compatibility issue with agent — try model: 'gemini-1.5-flash'");
      }
    }
  }
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});