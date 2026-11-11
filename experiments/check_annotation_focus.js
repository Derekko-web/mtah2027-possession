// Playwright CLI callback. Requires synthetic annotation QA on 127.0.0.1:8766.
async (originalPage) => {
  const page = await originalPage.context().newPage();
  try {
  await page.goto("http://127.0.0.1:8766");
  await page.getByRole("heading", {name:"Clip 01",exact:true}).waitFor();
  if (await page.locator("#source-frame").textContent() !== "Source frame 0") throw new Error("Use only the synthetic QA fixture");
  await page.waitForFunction(() => document.querySelector("video").readyState >= 2);
  await page.getByRole("button", {name:"Next frame",exact:true}).click();
  await page.waitForFunction(() => document.querySelector("#current-frame").textContent === "001" && !document.querySelector("video").seeking);
  await page.getByRole("textbox", {name:"Annotator",exact:true}).fill("Synthetic unsaved draft");
  const stage = page.locator(".video-stage");
  const bounds = await stage.boundingBox();
  // Compare the picture, excluding rounded-border compositing antialiasing.
  const shot = {clip:{x:bounds.x+16,y:bounds.y+16,width:bounds.width-32,height:bounds.height-32},animations:"disabled"};
  const before = await page.screenshot({...shot,path:"output/playwright/focus-before.png"});
  const expectedFrame = await page.locator("#current-frame").textContent();
  // Controlled renderer fault: a native video surface can stop being painted
  // while the DOM and currentTime remain intact. Exercise that exact visible
  // failure without requiring a particular host compositor or real footage.
  await page.evaluate(() => {
    window.dispatchEvent(new Event("blur"));
    document.querySelector("video").style.visibility = "hidden";
  });
  const away = await page.screenshot({...shot,path:"output/playwright/focus-away.png"});
  if (!before.equals(away)) throw new Error("BLACK FRAME REGRESSION: losing the native video surface replaces the paused picture ");
  await page.evaluate(() => { window.dispatchEvent(new Event("focus")); });
  await page.waitForFunction(() => !document.querySelector("video").seeking);
  const restored = await page.screenshot({...shot,path:"output/playwright/focus-restored.png"});
  if (!before.equals(restored)) throw new Error("Focus return did not preserve the paused picture");
  if (await page.locator("#current-frame").textContent() !== expectedFrame) throw new Error("Focus return changed the annotation frame");
  if (await page.getByRole("textbox", {name:"Annotator",exact:true}).inputValue() !== "Synthetic unsaved draft") throw new Error("Focus return lost the unsaved draft");
  await page.evaluate(() => { document.querySelector("video").style.visibility = ""; });
  // A manual control must also recover if the host misses its focus event.
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await page.getByRole("button", {name:"Next frame",exact:true}).click();
  await page.waitForFunction(() => document.querySelector("#current-frame").textContent === "002" && !document.querySelector("video").seeking);
  await page.getByRole("button", {name:"Play",exact:true}).click();
  await page.waitForFunction(() => Number(document.querySelector("#current-frame").textContent) >= 5);
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  const heldFrame = await page.locator("#current-frame").textContent();
  if (!(await page.evaluate(() => document.querySelector("video").paused && !document.querySelector("#paused-frame").hidden))) throw new Error("Playback did not pause on focus loss");
  for (let i=0;i<5;i++) {
    await page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await page.waitForFunction(() => !document.querySelector("video").seeking);
    if (await page.locator("#current-frame").textContent() !== heldFrame) throw new Error("Focus cycle moved the held frame");
    await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  }
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await page.waitForFunction(() => !document.querySelector("video").seeking);
  if (!(await page.evaluate(() => document.querySelector("video").paused))) throw new Error("Playback resumed without the user's action");
  return {paused_picture_preserved:true,frame_preserved:true,draft_preserved:true,manual_recovery:true,playback_pauses:true,repeated_focus_cycles:5,renderer_fault_injected:true};
  } finally { await page.close({runBeforeUnload:false}); }
}
