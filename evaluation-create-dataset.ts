import {Client} from "langsmith";
import 'dotenv/config';
import * as fs from 'fs';
import * as readline from 'readline';

const EVALUATE_DATASET = 'DocuAI Dataset';
const DATASET_FILE_PATH = './docuai-dataset.jsonl';
const DATASET_DESCRIPTION = 'DocuAI dataset in LangSmith.';

async function loadExamplesFromJSONL(filePath: string, datasetId: string) {
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    const examples: any[] = [];

    for await (const line of rl) {
        if (line.trim()) {
            try {
                const obj = JSON.parse(line);
                examples.push({
                    inputs: {question: obj.question},
                    outputs: {answer: obj.answer},
                    dataset_id: datasetId,
                });
            } catch (err) {
                console.error('Invalid JSONL line:', line);
            }
        }
    }

    return examples;
}

(async () => {
    // Define example input and reference output pairs that you'll use to evaluate your app

    const client = new Client();

    const dataset = await client.createDataset(EVALUATE_DATASET, {
        description: DATASET_DESCRIPTION,
    });

    // Create inputs and reference outputs
    // const examples = [
    //     {
    //         inputs: {question: "Which country is Mount Kilimanjaro located in?"},
    //         outputs: {answer: "Mount Kilimanjaro is located in Tanzania."},
    //         dataset_id: dataset.id,
    //     },
    //     {
    //         inputs: {question: "What is Earth's lowest point?"},
    //         outputs: {answer: "Earth's lowest point is The Dead Sea."},
    //         dataset_id: dataset.id,
    //     },
    // ];
    const examples = await loadExamplesFromJSONL(DATASET_FILE_PATH, dataset.id);

    // Add examples to the dataset
    await client.createExamples(examples);
})();