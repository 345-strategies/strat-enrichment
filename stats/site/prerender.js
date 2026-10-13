// Fill the empty containers of a built page with what its script renders on first load, so
// crawlers and agents that don't run JavaScript get the default tab as plain HTML.
// No dependencies: every view is an HTML string, so a stub document is enough.
// Usage: node prerender.js <page.html>
const fs = require('fs'), vm = require('vm');
const f = process.argv[2]; let src = fs.readFileSync(f, 'utf8');
const js = src.slice(src.lastIndexOf('<script>') + 8, src.lastIndexOf('</script>'));
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const els = {};
const el = () => { const o = { innerHTML: '' }; Object.defineProperty(o, 'textContent', { set(v) { o.innerHTML = esc(v) } }); return o };
const ctx = { document: { querySelector: s => (els[s] ??= el()) }, location: { hash: '' }, console,
  localStorage: { getItem: () => null, setItem() {}, clear() {} } };
vm.createContext(ctx);
vm.runInContext(js + "\n;setTab('overview');", ctx);
const slots = [['<h1 id="h1"></h1>', '#h1', 'h1 id="h1"', 'h1'], ['<div class="sub" id="sub"></div>', '#sub', 'div class="sub" id="sub"', 'div'],
  ['<div class="types" id="types"></div>', '#types', 'div class="types" id="types"', 'div'], ['<nav id="nav"></nav>', '#nav', 'nav id="nav"', 'nav'],
  ['<main id="view"></main>', '#view', 'main id="view"', 'main']];
for (const [empty, sel, open, tag] of slots) {
  if (!src.includes(empty)) throw new Error('missing ' + empty);
  src = src.replace(empty, () => `<${open}>${els[sel].innerHTML}</${tag}>`);
}
fs.writeFileSync(f, src); console.log('prerendered', f, src.length);
