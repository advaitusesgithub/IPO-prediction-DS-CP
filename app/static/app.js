"use strict";
const $ = selector => document.querySelector(selector);
const money = value => new Intl.NumberFormat("en-IN", {style:"currency",currency:"INR",maximumFractionDigits:2}).format(value);
const percent = value => new Intl.NumberFormat("en-IN", {style:"percent",maximumFractionDigits:1}).format(value);
let catalog = [], current = null, requestNumber = 0;

function message(text, loading = false) { $("#form-message").textContent = text; $("#form-message").classList.toggle("loading", loading); }
function suggestionList(items) {
  const box = $("#suggestions"); box.replaceChildren(); box.hidden = !items.length;
  items.forEach(item => {const button = document.createElement("button"); button.type="button"; button.textContent=item.name;
    button.addEventListener("click", () => {$("#ipo-name").value=item.name; predict(item.id);}); box.append(button);});
}
function renderList(filter = "") {
  const list = $("#historical-list"); list.replaceChildren();
  const available = catalog.filter(ipo => ipo.available);
  const filtered = available.filter(ipo => `${ipo.name} ${ipo.symbol} ${ipo.year}`.toLowerCase().includes(filter.toLowerCase()));
  if (!filtered.length) {const p=document.createElement("p");p.className="muted list-loading";p.textContent="No matching IPOs in this collection.";list.append(p);}
  filtered.forEach(ipo => {const button=document.createElement("button");button.type="button";button.className="historical-item";button.setAttribute("role","listitem");
    const content=document.createElement("span");content.textContent=ipo.name;const detail=document.createElement("small");detail.textContent=`${ipo.year}${ipo.symbol ? ` · ${ipo.symbol}` : ""}`;content.append(detail);
    const arrow=document.createElement("span");arrow.className="arrow";arrow.textContent="↗";arrow.setAttribute("aria-hidden","true");button.append(content,arrow);
    button.addEventListener("click",()=>{$("#ipo-name").value=ipo.name;predict(ipo.id);});list.append(button);});
  const years=available.map(ipo=>ipo.year);
  $("#historical-count").textContent=`${filtered.length} ${filter ? "matching" : "supported"} IPOs · ${Math.min(...years)}–${Math.max(...years)} collection`;
}
async function predict(name) {
  name=name.trim(); const id=++requestNumber;
  current=null; $("#result").hidden=true; $("#empty-state").hidden=false;
  suggestionList([]);
  if (!name) {$("#predict-button").disabled=false;$("#predict-button").textContent="Predict ↗";message("Enter an IPO name, or choose one from the historical collection.");$("#ipo-name").focus();return;}
  $("#predict-button").disabled=true;$("#predict-button").textContent="Predicting…";
  message("Finding your IPO and preparing its outlook…",true);$("#result").setAttribute("aria-busy","true");
  try {
    const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});
    const data=await response.json(); if(id!==requestNumber)return;
    if (!response.ok) {message(data.error || "We couldn’t load this prediction. Please try again.");suggestionList(data.suggestions || []);return;}
    current=data;message("");$("#empty-state").hidden=true;$("#result").hidden=false;
    $("#result-title").textContent=data.ipo.name;
    $("#result-meta").textContent=`${data.ipo.symbol ? `${data.ipo.symbol} · ` : ""}${data.ipo.venue} · ${data.ipo.year} listing`;
    $("#source-badge").textContent=data.source.label;$("#source-badge").classList.toggle("training",data.source.kind==="training");
    $("#predicted-price").textContent=money(data.prediction.opening_price);
    $("#predicted-probability").textContent=percent(data.prediction.positive_probability);
    $("#predicted-signal").textContent=data.prediction.signal;$("#predicted-signal").classList.toggle("skip",data.prediction.signal==="Skip");
    $("#price-band").textContent=`${money(data.prediction.band_low)} – ${money(data.prediction.band_high)}`;
    $("#source-description").textContent=`${data.source.description} Decision threshold: ${percent(data.prediction.threshold || .5)}.`;
    $("#actual-result").hidden=true;$("#actual-result").replaceChildren();$("#actual-button").textContent="See actual listing outcome ↗";
    $("#result-title").focus({preventScroll:true});
  } catch(error) {if(id===requestNumber)message("We couldn’t connect to the prediction service. Please try again.");}
  finally {if(id===requestNumber){$("#predict-button").disabled=false;$("#predict-button").textContent="Predict ↗";$("#result").removeAttribute("aria-busy");}}
}
$("#prediction-form").addEventListener("submit",event=>{event.preventDefault();predict($("#ipo-name").value);});
document.querySelectorAll("[data-example]").forEach(button=>button.addEventListener("click",()=>{$("#ipo-name").value=button.dataset.example;$("#ipo-name").focus();}));
$("#historical-filter").addEventListener("input",event=>renderList(event.target.value));
$("#actual-button").addEventListener("click",async()=>{
  if(!current)return;const selected=current.ipo.id, box=$("#actual-result");
  if(!box.hidden){box.hidden=true;$("#actual-button").textContent="See actual listing outcome ↗";return;}
  try {const response=await fetch(`/api/outcome/${encodeURIComponent(selected)}`);if(!response.ok)throw new Error();const actual=await response.json();if(!current || current.ipo.id!==selected)return;
    box.replaceChildren();[`${current.ipo.name} · recorded outcome`,`Actual opening: ${money(actual.opening_price)} · Final issue price: ${money(actual.issue_price)}`,`Actual listing gain: ${percent(actual.listing_gain)}. These are evaluation outcomes, not prediction inputs.`].forEach(line=>{const span=document.createElement("span");span.textContent=line;box.append(span);});box.hidden=false;$("#actual-button").textContent="Hide actual outcome ↑";
  }catch(error){message("The actual outcome could not be loaded. Please try again.");}
});
$("#download-result").addEventListener("click",()=>{if(!current)return;
  const escape=value=>`"${String(value).replaceAll('"','""')}"`;
  const rows=[["IPO","Estimated opening price (INR)","Positive-gain probability","Signal","Prediction basis"],[current.ipo.name,current.prediction.opening_price,current.prediction.positive_probability,current.prediction.signal,current.source.label]];
  const url=URL.createObjectURL(new Blob([rows.map(row=>row.map(escape).join(",")).join("\r\n")],{type:"text/csv;charset=utf-8;"}));const a=document.createElement("a");a.href=url;a.download="ipo_prediction.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
function showAbout(){$("#about-dialog").showModal();}
$("#about-button").addEventListener("click",showAbout);$("#footer-about").addEventListener("click",showAbout);$("#close-about").addEventListener("click",()=>$("#about-dialog").close());
$("#about-dialog").addEventListener("click",event=>{if(event.target===$("#about-dialog")){const rect=event.target.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.target.close();}});
async function initialize(){
  try {const response=await fetch("/api/ipos");if(!response.ok)throw new Error();catalog=(await response.json()).ipos;renderList();}
  catch(error){$("#historical-list").textContent="The collection couldn’t load. Refresh the page to try again.";}
  try {const response=await fetch("/api/summary");if(!response.ok)throw new Error();const summary=await response.json();$("#about-mae").textContent=money(summary.regression.mae_inr);$("#about-accuracy").textContent=percent(summary.classification.accuracy);let note=`Original 2025 evaluation: on 103 IPOs, the classifier predicted Apply every time. Its ${percent(summary.classification.accuracy)} accuracy equals always Apply; useful Skip detection and 90% accuracy have not been demonstrated. Announced cap alone had a lower price MAE (${money(summary.cap_baseline.mae_inr)}).`;if(summary.refinement){const r=summary.refinement, price=r.metrics.find(m=>m.method==='refined_v2'&&m.task==='regression'), clf=r.metrics.find(m=>m.method==='refined_v2'&&m.task==='classification');note+=` A separate ${r.external_rows}-IPO 2026 audit produced price MAE ${money(price.mae_inr)} and classification accuracy ${percent(clf.accuracy)}. It also found no useful Skip detection. These are different test periods.`;}$("#model-limitation").textContent=note;}
  catch(error){$("#model-limitation").textContent="Performance details are currently unavailable. Predictions are historical estimates, not guaranteed outcomes.";}
}
initialize();
