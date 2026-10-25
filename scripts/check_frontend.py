"""Exercise the real browser flow against an already running local app."""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8511')
args=parser.parse_args()
out=root/'outputs/ui'
out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    env=dict(os.environ)
    local_libs=root/'.runtime/browser-libs/lib/usr/lib/x86_64-linux-gnu'
    if local_libs.exists():
        env['LD_LIBRARY_PATH']=str(local_libs)+':'+env.get('LD_LIBRARY_PATH','')
    browser=p.chromium.launch(headless=True,env=env)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(args.url)
    expect(page.locator('.historical-item').first).to_be_visible(timeout=60000)
    expect(page.locator('#empty-state')).to_be_visible()
    expect(page.locator('#result')).to_be_hidden()
    page.screenshot(path=str(out/'desktop_initial.png'),full_page=True)
    page.get_by_role('button',name='Predict',exact=False).click()
    expect(page.locator('#form-message')).to_contain_text('Enter an IPO name')
    page.locator('#ipo-name').fill('Hyundai Motor India')
    page.locator('#predict-button').click()
    expect(page.locator('#result-title')).to_contain_text('Hyundai',timeout=60000)
    expect(page.locator('#predicted-price')).to_contain_text('₹')
    expect(page.locator('.prediction-table')).to_be_visible()
    page.locator('#result').evaluate('(el) => el.getAnimations().forEach(animation => animation.finish())')
    page.screenshot(path=str(out/'desktop_prediction.png'),full_page=True)
    page.locator('#actual-button').click()
    expect(page.locator('#actual-result')).to_be_visible()
    with page.expect_download() as download:
        page.locator('#download-result').click()
    assert download.value.suggested_filename=='ipo_prediction.csv'
    page.locator('#ipo-name').fill('Not A Real Issuer QZXV 999')
    page.locator('#predict-button').click()
    expect(page.locator('#form-message')).to_contain_text('couldn’t find')
    expect(page.locator('#result')).to_be_hidden()
    page.locator('#ipo-name').fill('India')
    page.locator('#predict-button').click()
    expect(page.locator('#form-message')).to_contain_text('A few IPOs')
    expect(page.locator('#suggestions button').first).to_be_visible()
    page.locator('#historical-filter').fill('NSDL')
    page.locator('.historical-item').first.click()
    expect(page.locator('#result-title')).to_contain_text('National Securities',timeout=60000)
    expect(page.locator('#source-badge')).to_contain_text('Held-out')
    page.locator('#ipo-name').fill('Radiant Cash')
    page.locator('#ipo-name').press('Enter')
    expect(page.locator('#source-badge')).to_contain_text('Training illustration',timeout=60000)
    expect(page.locator('#source-description')).to_contain_text('not an independent')
    page.locator('#about-button').click()
    expect(page.get_by_role('dialog')).to_be_visible()
    expect(page.locator('#model-limitation')).to_contain_text('68')
    page.keyboard.press('Escape')
    expect(page.get_by_role('dialog')).to_be_hidden()
    page.locator('#ipo-name').fill('Bharat Coking Coal')
    page.locator('#predict-button').click()
    expect(page.locator('#source-badge')).to_contain_text('2026 audit',timeout=60000)
    expect(page.locator('#source-description')).to_contain_text('45%')
    mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1,is_mobile=True,has_touch=True)
    mobile.goto(args.url)
    expect(mobile.locator('.historical-item').first).to_be_visible(timeout=60000)
    mobile.locator('#ipo-name').fill('NSDL')
    mobile.locator('#predict-button').click()
    expect(mobile.locator('#result-title')).to_contain_text('National Securities',timeout=60000)
    assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    mobile.locator('#result').evaluate('(el) => el.getAnimations().forEach(animation => animation.finish())')
    mobile.screenshot(path=str(out/'mobile_prediction.png'),full_page=True)
    assert not errors,errors
    browser.close()
summary={'desktop_and_mobile_verified':True,'name_enter_predict_flow':True,
    'historical_picker':True,'ambiguous_and_unknown_names':True,'training_label':True,
    'csv_download':True,'actual_outcome_toggle':True,'model_details_dialog':True,
    'mobile_horizontal_overflow':False,'javascript_errors':errors}
(out/'browser_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
