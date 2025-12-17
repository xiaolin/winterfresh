import 'dotenv/config';
import OpenAI from 'openai';
import { speak } from './tts.ts';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function main() {
  const input = process.argv.slice(2).join(' ');
  if (!input) throw new Error('Usage: tsx src/chat.ts "hello"');

  const stream = await client.responses.stream({
    model: 'gpt-4.1-mini',
    input,
  });

  let full = '';
  stream.on('response.output_text.delta', (e) => {
    process.stdout.write(e.delta);
    full += e.delta;
  });

  await stream.finalResponse();
  process.stdout.write('\n');

  await speak(full);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
