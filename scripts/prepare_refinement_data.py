"""Import only three small existing sources and construct unused 2026 snapshots."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ipo_prediction.data import load_dataset
from ipo_prediction.refinement_models import CONTEXT


def number(value):
    try:return float(str(value).replace(',','').strip())
    except (ValueError,TypeError):return np.nan


def amount(text,cap):
    money=re.search(r'(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(million|crore|lakh)',text,re.I)
    if money:return number(money[1])*{'million':.1,'crore':1.,'lakh':.01}[money[2].lower()]
    shares=re.search(r'([\d,]+)\s*(?:equity\s*)?shares',text,re.I)
    return number(shares[1])*cap/1e7 if shares else np.nan


def offer_size(text,cap):
    text=' '.join(str(text).split())
    fresh=re.search(r'fresh issue(.*?)(?=offer (?:for|of) sale|including|$)',text,re.I)
    ofs=re.search(r'offer (?:for|of) sale(.*?)(?=including|$)',text,re.I)
    f=amount(fresh[1],cap) if fresh else np.nan
    o=amount(ofs[1],cap) if ofs else np.nan
    if fresh and not ofs:return f
    if ofs and not fresh:return o
    if np.isfinite(f) and np.isfinite(o):return f+o
    return np.nan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,help='Original Datasets folder, needed only for initial import')
    args=parser.parse_args()
    base=ROOT/'data/refinement';base.mkdir(parents=True,exist_ok=True)
    sources={'profiles.csv':'IPO_AI_mainboard_ipos.csv','cohort.csv':'IPO_performance_verification/data/verified/ipo_cohort_verified.csv',
             'nifty.csv':'Nifty_50_historical_GitHub.csv'}
    for local,original in sources.items():
        if not (base/local).exists():
            if args.source_dir is None:raise ValueError('Supply --source-dir for initial local import')
            shutil.copy2(args.source_dir/original,base/local)
    pd.DataFrame([{'file':k,'original_source':v,'sha256':hashlib.sha256((base/k).read_bytes()).hexdigest()} for k,v in sources.items()]).to_csv(base/'source_manifest.csv',index=False)
    profiles=pd.read_csv(base/'profiles.csv')
    cohort=pd.read_csv(base/'cohort.csv')
    new=cohort.loc[cohort.board_type.eq('Main Board')&cohort.listing_year.eq(2026)].copy()
    old=load_dataset()
    market_raw=pd.read_csv(base/'nifty.csv')
    market=pd.DataFrame({'date':pd.to_datetime(market_raw.Date,format='%m/%d/%Y'),
                         'nifty_close':market_raw.Price.map(number)}).drop_duplicates('date').sort_values('date').reset_index(drop=True)
    for n,maximum in [(5,12),(21,42),(63,120)]:
        market[f'nifty_return_{n}d']=market.nifty_close.pct_change(n,fill_method=None).where(market.date.sub(market.date.shift(n)).dt.days.le(maximum))
    market['nifty_volatility_21d']=np.log(market.nifty_close).diff().rolling(21,min_periods=21).std(ddof=1)*np.sqrt(252)
    # Rebuild old-period market transforms independently, without modifying inputs.
    prior=market.set_index('date')
    for _,row in old.iterrows():
        recent=market.loc[market.date.lt(row.s0_cutoff_at.tz_localize(None).normalize())].iloc[-1]
        for col in ['nifty_return_5d','nifty_return_21d','nifty_return_63d','nifty_volatility_21d']:
            if not np.isclose(row[col],recent[col],atol=1e-12,rtol=1e-12):raise ValueError('Old market feature mismatch')
    history=old[['ipo_id','listing_date','label_available_at','target_listing_gain_open']].copy()
    hnew=pd.DataFrame({'ipo_id':new.issue_id,'listing_date':pd.to_datetime(new.listing_date),
        'label_available_at':pd.to_datetime(new.listing_date).dt.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=23,minutes=59,seconds=59),
        'target_listing_gain_open':new.listing_open_inr/new.issue_price_inr-1})
    history=pd.concat([history.reset_index(drop=True),hnew],ignore_index=True)
    rows=[];audit=[]
    for _,label in new.iterrows():
        matches=profiles.loc[profiles.symbol.eq(label.nse_symbol)]
        reason='profile unavailable'
        if len(matches)==1:
            profile=matches.iloc[0];opening=pd.to_datetime(profile.ipo_start,errors='coerce')
            reason='market history ends before IPO opening'
            if pd.notna(opening) and opening<=market.date.max():
                band=re.findall(r'[\d,]+(?:\.\d+)?',str(profile.nse_price_range))
                reason='original NSE band or lot unavailable'
                if len(band)>=2 and number(profile.nse_bid_lot)>0:
                    low,cap=map(number,band[:2]);cutoff=opening.tz_localize('Asia/Kolkata')+pd.Timedelta(hours=9)
                    available=market.loc[market.date.lt(opening)]
                    latest=available.iloc[-1];age=(opening-latest.date).days
                    reason='stale market observation'
                    if 0<age<=7:
                        reason='source listing date disagreement'
                        if pd.to_datetime(profile.listing_date)==pd.to_datetime(label.listing_date):
                            reason='eligible'
                            mature=history.loc[history.label_available_at.lt(cutoff)]
                            record={'ipo_id':str(label.issue_id),'company_name':label.company_name,'symbol':label.nse_symbol,
                                's0_cutoff_at':cutoff,'listing_date':pd.to_datetime(label.listing_date),
                                'label_available_at':pd.Timestamp(label.listing_date,tz='Asia/Kolkata')+pd.Timedelta(hours=23,minutes=59,seconds=59),
                                'listing_open_inr':label.listing_open_inr,'final_issue_price_inr':label.issue_price_inr,
                                'target_positive_listing_gain':int(label.listing_open_inr>label.issue_price_inr),
                                'target_listing_gain_open':label.listing_open_inr/label.issue_price_inr-1,
                                'band_high_inr':cap,'band_low_inr':low,'band_width_fraction':(cap-low)/cap,
                                'lot_application_at_cap_inr':cap*number(profile.nse_bid_lot),
                                'log1p_planned_offer_upper_cr':np.log1p(offer_size(profile.nse_issue_size_text,cap)),
                                'market_source_date':latest.date,'market_source_age_days':age,
                                'latest_history_label':mature.label_available_at.max()}
                            for col in ['nifty_return_5d','nifty_return_21d','nifty_return_63d','nifty_volatility_21d']:record[col]=latest[col]
                            for days in [30,90]:
                                h=mature.loc[mature.listing_date.ge(opening-pd.Timedelta(days=days))]
                                record[f'recent_listed_ipo_count_{days}d']=len(h)
                                record[f'recent_median_listing_gain_{days}d']=h.target_listing_gain_open.median()
                            rows.append(record)
        audit.append({'ipo_id':label.issue_id,'company_name':label.company_name,'status':reason})
    frame=pd.DataFrame(rows).sort_values(['s0_cutoff_at','ipo_id']).reset_index(drop=True)
    assert len(frame)>0 and not frame.ipo_id.duplicated().any()
    assert set(frame.ipo_id).isdisjoint(old.index)
    assert frame.latest_history_label.lt(frame.s0_cutoff_at).all()
    assert frame.market_source_age_days.between(1,7).all()
    assert np.isfinite(frame[CONTEXT].to_numpy(dtype=float)).all()
    frame.to_parquet(base/'external_2026.parquet',index=False)
    frame.to_csv(base/'external_2026.csv',index=False)
    pd.DataFrame(audit).to_csv(base/'external_eligibility.csv',index=False)
    market.to_parquet(base/'nifty_transforms.parquet',index=False)
    print('Prepared fresh 2026 IPOs:',len(frame),'out of',len(new))
    print(pd.DataFrame(audit).status.value_counts().to_string())


if __name__=='__main__':main()
