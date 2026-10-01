"""Fixed, authored short-context contrast audit; never used for training.

These are constructed candidate sets, not real Mozc N-best or field logs.
Each pair has identical reading and candidate order, with opposite golds.
Only the left context is supplied. Results are diagnostic, not a production
accuracy estimate. All text was authored for this test (CC0-1.0).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.rerank.cpu_optimization import load_runtime

PAIRS = [
    ("きしゃ", "記者", "汽車", "新聞社で取材を担当する", "蒸気機関で走る昔の"),
    ("きかん", "期間", "器官", "申し込みを受け付けている", "心臓や肺などの人体の"),
    ("こうか", "効果", "硬貨", "薬を飲んで症状が改善する", "自動販売機に入れる百円の"),
    ("かいとう", "回答", "解凍", "アンケートの質問への", "冷凍した肉を電子レンジで"),
    ("せいか", "成果", "生花", "努力を重ねて得られた研究の", "仏壇に供える造花ではない"),
    ("しょうか", "消化", "消火", "胃や腸で食べ物を", "燃えている火を水で"),
    ("こうえん", "公園", "講演", "子どもがブランコで遊ぶ近所の", "大学の教授が登壇して行う"),
    ("きょうかい", "協会", "教会", "業界の企業が加盟する日本自動車", "キリスト教の信者が礼拝に集まる"),
    ("しこう", "思考", "施行", "難しい問題を解くために深く", "新しい法律が来月から"),
    ("きせい", "規制", "帰省", "事故を防ぐため道路の交通を", "お盆にふるさとの実家へ"),
    ("ほしょう", "保証", "補償", "故障した商品を無料で修理するメーカーの", "事故で生じた損害をお金で"),
    ("たいしょう", "対象", "対称", "調査を行う相手を示す調査", "左右を鏡に映したような図形の"),
    ("かてい", "家庭", "仮定", "子どもが両親と暮らす温かい", "もし雨が降るとした場合の"),
    ("せいさん", "生産", "精算", "工場で製品を大量に", "立て替えた旅費を経理で"),
    ("しゅうりょう", "終了", "修了", "イベントが予定時刻で", "大学院の課程をすべて履修して"),
    ("きょうこう", "強行", "恐慌", "反対を押し切って計画を", "大不況に陥った昭和の経済"),
    ("かんせい", "完成", "歓声", "建設工事が終わり建物が", "ゴールが決まると観客席から大きな"),
    ("いどう", "移動", "異動", "荷物を隣の部屋へ", "人事発令により営業部へ"),
    ("けっさい", "決済", "決裁", "クレジットカードで代金を", "稟議書に社長の承認をもらうための"),
    ("こうせい", "構成", "校正", "複数の章からなる本の", "誤字脱字がないか原稿を"),
    ("かいしゅう", "回収", "改修", "使い終わったペットボトルを", "古くなった橋を補強するため"),
    ("はんこう", "反抗", "犯行", "親の言うことに逆らう子どもの", "警察は強盗事件の"),
    ("かんしん", "関心", "感心", "科学のニュースに興味や", "子どもの見事な演奏に"),
    ("いぎ", "意義", "異議", "この研究を行う目的や社会的な", "判決に納得できず裁判所に"),
    ("きこう", "気候", "機構", "一年を通して暖かい沖縄の", "歯車を組み合わせて動く時計の"),
    ("しじ", "指示", "支持", "上司から作業の手順について", "選挙でその候補者を"),
    ("かいほう", "開放", "解放", "誰でも入れるよう体育館を一般に", "人質を無事に"),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--tau",type=float,default=2.5)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    runtime=load_runtime(args.runtime)
    scorer=runtime.OrtScorer(Path(args.model),Path(args.tokenizer),128,4)
    results=[];paired=final_paired=0
    for reading,a,b,ca,cb in PAIRS:
        pair=[]
        for gold,context in ((a,ca),(b,cb)):
            req={"reading":reading,"context_prev":context,"nbest":[a,b]}
            response=runtime.rerank(req,scorer,args.tau,30)
            result={"reading":reading,"gold":gold,"context":context,
                "neural_top1":response["rerank_top1"],"final_top1":response["final_top1"],
                "margin":response.get("margin",0),"guard_skip":response["guard_skip"],
                "neural_correct":response["rerank_top1"]==gold,
                "final_correct":response["final_top1"]==gold}
            pair.append(result);results.append(result)
        paired+=all(r["neural_correct"] for r in pair)
        final_paired+=all(r["final_correct"] for r in pair)
    report={"cases":len(results),"pairs":len(PAIRS),"mozc_constructed_hit1":.5,
        "neural_hit1":sum(r["neural_correct"] for r in results)/len(results),
        "final_hit1":sum(r["final_correct"] for r in results)/len(results),
        "pair_both_correct":paired,"final_pair_both_correct":final_paired,
        "model":args.model,"tau":args.tau,"kind":"authored_diagnostic_only",
        "used_for_training":False,"results":results}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)


if __name__=="__main__":main()
