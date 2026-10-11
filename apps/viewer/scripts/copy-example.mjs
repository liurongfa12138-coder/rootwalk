import { cp, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const viewerRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.resolve(viewerRoot, "../../examples/bank-recorded-run");
const destination = path.resolve(viewerRoot, "public/example-run");

await mkdir(destination, { recursive: true });
await cp(source, destination, { recursive: true, force: true });
console.log(`Copied recorded RunBundle to ${path.relative(viewerRoot, destination)}`);
