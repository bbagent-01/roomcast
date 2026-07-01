// Stores per-photo refine notes from the apartment review into the roomcast_refine KV (binding REFINE).
// Read via Cloudflare API: GET /accounts/{acct}/storage/kv/namespaces/{ns}/keys  +  /values/{key}
export async function onRequestPost(context) {
  const { request, env } = context;
  try {
    const body = await request.json();
    const room = (body.room || "unknown").toString().slice(0, 80);
    const note = (body.note || "").toString().slice(0, 2000);
    const round = (body.round || "").toString().slice(0, 20);
    if (!note.trim()) return json({ ok: false, error: "empty" }, 400);
    const at = new Date().toISOString();
    let stored = false;
    if (env.REFINE) {
      await env.REFINE.put(`${at}__${room}__${round}`, JSON.stringify({ room, note, round, at }));
      stored = true;
    }
    return json({ ok: true, stored });
  } catch (e) {
    return json({ ok: false, error: String(e) }, 500);
  }
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
}
