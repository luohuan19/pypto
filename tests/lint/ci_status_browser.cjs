const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const number = process.env.PROOF_PR;
  if (!/^[1-9][0-9]*$/.test(number)) throw new Error('Invalid PR number');
  fs.mkdirSync('ci-display-proof', {recursive: true});
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1200}});
  await page.goto(`https://github.com/luohuan19/pypto/pull/${number}`, {waitUntil: 'domcontentloaded'});
  await page.waitForTimeout(15000);
  fs.writeFileSync('ci-display-proof/pr.html', await page.content());
  fs.writeFileSync('ci-display-proof/pr.txt', await page.locator('body').innerText());
  await page.screenshot({path: 'ci-display-proof/pr.png', fullPage: true});
  const links = await page.locator('a').evaluateAll(as => as.map(a => ({text: a.textContent.trim(), href: a.href})).filter(a => a.href.includes('/actions/runs/') || a.text.includes('full /')));
  console.log(JSON.stringify(links));
  fs.writeFileSync('ci-display-proof/links.json', JSON.stringify(links, null, 2));
  await browser.close();
})();
