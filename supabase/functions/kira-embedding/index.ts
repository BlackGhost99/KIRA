// Embeddings dans Supabase, sans modèle à installer sur les appareils ni sur Render.
// Cette fonction expose un endpoint serveur à serveur, protégé par un secret dédié.
const model = new Supabase.ai.Session("gte-small");

async function equalSecrets(a: string, b: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const hashes = await Promise.all([a, b].map(s => crypto.subtle.digest("SHA-256", encoder.encode(s))));
  const left = new Uint8Array(hashes[0]), right = new Uint8Array(hashes[1]);
  let difference = 0;
  for (let i = 0; i < left.length; i++) difference |= left[i] ^ right[i];
  return difference === 0;
}

Deno.serve(async (request: Request) => {
  const secret = Deno.env.get("KIRA_EMBEDDING_SECRET") || "";
  const bearer = request.headers.get("authorization") || "";
  if (!secret || !await equalSecrets(bearer, "Bearer " + secret)) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (request.method !== "POST") return Response.json({ error: "Method not allowed" }, { status: 405 });
  // Limite réellement appliquée au flux, même sans Content-Length.
  const reader = request.body?.getReader();
  if (!reader) return Response.json({ error: "Empty body" }, { status: 400 });
  let total = 0;
  const chunks: Uint8Array[] = [];
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > 100_000) {
        await reader.cancel();
        return Response.json({ error: "Body too large" }, { status: 413 });
      }
      chunks.push(value);
    }
    const buffer = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
    const body = JSON.parse(new TextDecoder().decode(buffer));
    if (typeof body.input !== "string" || !body.input.trim() || body.input.length > 8000 || body.model !== "gte-small") {
      return Response.json({ error: "Invalid input or model" }, { status: 400 });
    }
    const embedding = await model.run(body.input, { mean_pool: true, normalize: true });
    return Response.json({ model: "gte-small", embedding: Array.from(embedding) });
  } catch {
    return Response.json({ error: "Embedding unavailable" }, { status: 503 });
  }
});
