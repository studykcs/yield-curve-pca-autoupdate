"""Detect anomalous days in the yield curve using PCA factor scores.

A day is flagged when its PC1 (level), PC2 (slope) or PC3 (curvature) score
exceeds a z-score threshold, where z is the day's score divided by that
component's own standard deviation (sqrt of its eigenvalue). Because the
components are orthogonal this is just a per-factor sigma check, not a
joint statistic.

Usage
-----
    python anomaly.py --from-db
    python anomaly.py --from-db --threshold 2.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from store import get_connection, load_yields
from pca import OUT, PCAResult, run_pca

PC_LABELS = {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}
ALERT_LOG = OUT / "alerts.json"

# Brief macro context for each anomaly date, keyed by ISO date (YYYY-MM-DD).
# Hand-compiled from known market history; not derived from the data itself.
MACRO_CONTEXT = {
    "2015-12-08": "유가 급락 지속, 12월 FOMC(첫 금리인상) 대기",
    "2016-01-05": "중국 증시 서킷브레이커 발동, 위안화 절하 우려로 시장 불안 시작",
    "2016-02-02": "BOJ 마이너스 금리 도입 여파, 유럽은행 신용우려 확산",
    "2016-06-24": "브렉시트(영국 EU 탈퇴) 국민투표 가결, 안전자산 선호 급등",
    "2016-11-09": "트럼프 미국 대통령 당선 확정, 리플레이션 트레이드로 금리 급등",
    "2017-10-10": "美 세제개편안 논의 진전, 연준 차기 의장 인선 관측",
    "2018-01-30": "인플레이션 우려에 국채 매도, 2월 초 변동성 쇼크(Volmageddon) 전조",
    "2018-05-29": "이탈리아 정치 위기(정부 구성 실패)로 안전자산 선호, 국채금리 급락",
    "2018-12-31": "12월 FOMC 인상 이후 증시 급락 지속, 안전자산 수요",
    "2019-12-30": "미중 1단계 무역합의 이후 연말 저유동성 장세",
    "2020-02-28": "코로나19 팬데믹 공포로 글로벌 증시 폭락, 연준 긴급 인하 기대",
    "2020-03-02": "코로나 확산 우려 지속, 연준 긴급 금리인하 기대 형성",
    "2020-03-04": "연준 긴급 50bp 인하(3/3) 여파, 코로나 확산 지속",
    "2020-03-05": "코로나19 팬데믹 공포 지속, 안전자산 쏠림",
    "2020-03-06": "사우디-러시아 OPEC+ 감산 합의 결렬 임박, 유가·증시 불안",
    "2020-03-09": "유가전쟁 발발(사우디-러시아), 뉴욕증시 서킷브레이커 발동",
    "2020-03-10": "유가 급락 이후 반발 매수, 극심한 변동성 지속",
    "2020-03-16": "연준 긴급 제로금리·QE 발표(3/15) 후에도 서킷브레이커 재발동",
    "2020-03-17": "국채시장 유동성 경색, 재무부·연준 유동성 공급 확대",
    "2020-03-18": "국채시장 기능 이상(유동성 위기), 연준 개입 확대",
    "2020-03-19": "글로벌 달러 유동성 경색 지속",
    "2020-03-20": "연준 스왑라인 확대 등 유동성 공급 지속",
    "2020-03-23": "연준 무제한 양적완화(QE) 발표, 회사채 매입 프로그램 신설",
    "2020-11-09": "화이자 코로나 백신 임상 효능 발표('백신 먼데이'), 경기민감주 순환매",
    "2021-02-25": "美 7년물 국채입찰 부진, 리플레이션 우려로 금리 급등(2월 국채발작)",
    "2021-11-26": "오미크론 변이 발견 발표, 글로벌 위험자산 급락(블랙프라이데이)",
    "2022-02-10": "1월 美 CPI 7.5% 서프라이즈(1982년來 최고), 3월 50bp 인상론 부각",
    "2022-02-22": "러시아, 우크라이나 돈바스 지역 독립 승인 및 군 진입(전쟁 전야)",
    "2022-03-02": "우크라이나 전쟁 속 파월 의장 의회 증언, 3월 인상 시사",
    "2022-03-21": "파월 의장, 필요시 50bp 인상 가능성 시사(NABE 연설)",
    "2022-03-22": "매파적 연준 발언 지속에 금리 재상승",
    "2022-03-25": "매파적 Fed 리프라이싱 지속, 장단기 금리차 축소",
    "2022-04-11": "4월 CPI 발표 앞둔 인플레이션 경계, 금리 역전 우려",
    "2022-06-10": "5월 CPI 8.6% 서프라이즈(41년래 최고), 75bp 인상론 급부상",
    "2022-06-13": "CPI 쇼크 여파로 국채금리 급등, 75bp 인상 기정사실화",
    "2022-06-14": "FOMC 전날, 75bp 인상 가능성 리프라이싱 지속",
    "2022-06-15": "연준 0.75%p 인상 단행(1994년 이후 최대폭)",
    "2022-06-21": "파월 의장 의회 증언, 인플레이션 파이팅 재확인",
    "2022-06-22": "긴축 장기화 우려 속 경기침체 논쟁 확산",
    "2022-07-01": "경기침체 공포 정점, ISM 제조업 지표 부진",
    "2022-07-12": "6월 CPI 발표 앞둔 포지셔닝, 100bp 인상설 부상",
    "2022-07-13": "6월 CPI 9.1%로 인플레이션 정점, 100bp 인상설 확산",
    "2022-08-02": "펠로시 대만 방문으로 미중 긴장 고조",
    "2022-08-05": "7월 비농업고용 52.8만명 서프라이즈, 긴축 장기화 우려",
    "2022-09-06": "유럽 에너지 위기(노르드스트림) 속 ECB 인상 경계",
    "2022-09-13": "8월 CPI 서프라이즈, S&P500 2020년 이후 최대 낙폭",
    "2022-09-21": "연준 3연속 75bp 인상, 매파적 점도표 공개",
    "2022-09-28": "英 미니예산발 길트채 위기, BOE 긴급 국채매입 개입",
    "2022-10-04": "OPEC+ 대규모 감산 발표 앞둔 단기 반등(숏커버링)",
    "2022-10-11": "9월 CPI 발표 앞둔 경계감, 국채금리 변동성 확대",
    "2022-10-13": "9월 CPI 서프라이즈 후 장중 극적 반전 랠리(역사적 변동성)",
    "2022-10-21": "WSJ, 연준 12월 인상 속도조절 시사 보도(Fed pivot 기대)",
    "2022-11-10": "10월 CPI 둔화 서프라이즈, 국채금리 급락·증시 급등",
    "2022-11-16": "CPI 서프라이즈 이후 되돌림, 연준 위원 발언 주시",
    "2022-12-27": "BOJ YCC 밴드 확대(12/20) 여파, 연말 저유동성 장세",
    "2023-01-06": "12월 고용보고서 임금상승률 둔화, 연준 속도조절 기대 강화",
    "2023-02-03": "1월 비농업고용 51.7만명 서프라이즈, 긴축 장기화 우려 재부각",
    "2023-03-07": "파월 의장 반기 의회 증언, 최종금리 상향·재가속 가능성 시사",
    "2023-03-10": "실리콘밸리은행(SVB) 파산, 금융시스템 불안 확산",
    "2023-03-13": "SVB·시그니처은행 연쇄 파산, 연준 BTFP 긴급 유동성 프로그램 가동",
    "2023-03-15": "크레디트스위스(CS) 위기 확산, 유럽 은행권 신용우려",
    "2023-03-16": "CS, 스위스중앙은행 유동성 지원 확보 후 반등",
    "2023-03-17": "은행위기 전염 지속, 대형은행 컨소시엄 300억달러 예금 지원",
    "2023-03-21": "FOMC(3/22) 결정 앞둔 포지셔닝",
    "2023-03-22": "연준 0.25%p 인상 단행(은행위기 속), 파월 긴축 종료 근접 시사",
    "2023-03-23": "퍼스트리퍼블릭은행 주가 급락, 은행위기 재확산 우려",
    "2023-03-24": "도이체방크 CDS 급등, 유럽 은행권 전염 우려",
    "2023-03-27": "First Citizens의 SVB 자산 인수로 은행위기 일단 진정",
    "2023-04-25": "퍼스트리퍼블릭은행 실적서 대규모 예금유출 확인, 은행위기 재점화",
    "2023-05-02": "퍼스트리퍼블릭은행 결국 파산, JP모건에 인수",
    "2023-05-04": "지역은행 스트레스(PacWest 등) 지속 및 부채한도 우려",
    "2023-05-16": "美 부채한도 협상(X-date 임박) 우려 고조",
    "2023-06-29": "1분기 GDP 수정치 발표, 통상적 월말 변동성",
    "2023-08-08": "무디스, 美 중소은행 다수 신용등급 강등",
    "2023-09-21": "연준 동결이나 매파적 점도표('higher for longer'), 장기금리 급등",
    "2023-09-25": "'higher for longer' 리프라이싱 지속, 정부 셧다운 우려",
    "2023-10-19": "파월 연설(NY 이코노믹클럽), 10년물 금리 5% 근접 속 시장 주시",
    "2023-11-02": "연준 2회 연속 동결, 비둘기파적 기자회견에 채권 랠리 시작",
    "2023-11-14": "10월 CPI 둔화 서프라이즈, 국채금리 급락(피벗 기대 재점화)",
    "2023-12-01": "파월, 긴축 종료 시사 발언에 금리인하 기대 형성",
    "2023-12-13": "연준 점도표에 2024년 세 차례 인하 시사('파월 피벗'), 금리 급락",
    "2024-02-02": "1월 고용 서프라이즈(35.3만명), 조기 금리인하 기대 후퇴",
    "2024-02-13": "1월 CPI 예상치 상회, 인하 기대 추가 후퇴",
    "2024-04-10": "3월 CPI 3개월 연속 서프라이즈, 6월 인하 기대 소멸",
    "2024-06-07": "5월 고용 서프라이즈, 인하 시점 재차 후퇴",
    "2024-08-02": "7월 고용 쇼크(Sahm 룰 발동), 경기침체 우려로 금리 급락(엔캐리 청산 서막)",
    "2024-08-06": "8월초 변동성 지속(VIX 급등, 엔캐리 트레이드 청산 여파)",
    "2024-10-04": "9월 고용 서프라이즈, 경기침체 우려 해소되며 금리 반등",
    "2025-04-03": "트럼프 '해방의 날' 상호관세 발표(4/2), 글로벌 무역전쟁 우려 급등",
    "2025-04-07": "관세발 시장 패닉 지속, 국채금리 변동성 확대",
    "2025-04-08": "관세 불확실성 속 국채시장 불안정, 유동성 우려",
    "2025-04-09": "트럼프, 상호관세 90일 유예 발표(국채금리 급등 이후)",
    "2025-04-10": "관세 유예 후 안도 랠리, 금리 되돌림",
    "2025-04-21": "연준 독립성 관련 논란 고조, 달러·국채 신뢰 우려",
    "2025-08-01": "7월 고용 부진 및 대규모 하향수정, BLS 국장 경질 논란",
    "2025-08-05": "고용 쇼크 여파 지속, 안전자산 수요",
    "2025-09-05": "8월 고용 둔화 확인, 9월 금리인하 기대 강화",
    "2026-06-17": "6월 FOMC(6/16-17), 인하 기대 축소·연내 동결 전망 우세",
    "2026-07-07": "美 30년물 국채금리 5% 상회, 무역적자 확대·나토 정상회의 대기",
    "2026-07-29": "FOMC 동결(9-3, 3명은 인상 주장하며 이견), 매파적 색채에 금리 급등",
}


def zscores(res: PCAResult, n_components: int = 3) -> pd.DataFrame:
    cols = [f"PC{i + 1}" for i in range(n_components)]
    vol = res.factor_vol_bp[:n_components]
    return res.scores[cols] / vol


def detect_anomalies(res: PCAResult, threshold: float = 3.0, n_components: int = 3) -> pd.DataFrame:
    """One row per (date, component) breach, sorted by date."""
    z = zscores(res, n_components)
    breached = z.abs() > threshold

    rows = [
        {
            "date": date,
            "component": pc,
            "label": PC_LABELS.get(pc, pc),
            "z_score": z.loc[date, pc],
            "move_bp": res.scores.loc[date, pc],
        }
        for date in z.index[breached.any(axis=1)]
        for pc in z.columns
        if breached.loc[date, pc]
    ]
    columns = ["date", "component", "label", "z_score", "move_bp"]
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def alert_new(anomalies: pd.DataFrame, log_path: Path = ALERT_LOG) -> pd.DataFrame:
    """Print and persist only the anomalies not already in the alert log.

    Re-running the pipeline recomputes the same historical anomalies every
    time, so without dedup every run would re-alert on old news. The log
    tracks (date, component) pairs already surfaced.
    """
    anomalies = anomalies.copy()
    anomalies["date"] = pd.to_datetime(anomalies["date"]).dt.strftime("%Y-%m-%d")

    seen = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    seen_keys = {(r["date"], r["component"]) for r in seen}

    is_new = ~anomalies.apply(lambda r: (r["date"], r["component"]) in seen_keys, axis=1)
    new = anomalies[is_new]

    if new.empty:
        print("No new anomalies since last check.")
        return new

    print(f"\n{len(new)} NEW anomal{'y' if len(new) == 1 else 'ies'} since last check:\n")
    for _, r in new.iterrows():
        print(f"  ALERT  {r['date']}  {r['label']:<10} z={r['z_score']:+.2f}  move={r['move_bp']:+.2f} bp")

    new_records = new[["date", "component", "label", "z_score", "move_bp"]].to_dict("records")
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for rec in new_records:
        rec["detected_at"] = now

    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(seen + new_records, indent=2), encoding="utf-8")
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold (default 3.0)")
    parser.add_argument("--reset-log", action="store_true", help="Clear the alert log before checking")
    args = parser.parse_args()

    if args.reset_log and ALERT_LOG.exists():
        ALERT_LOG.unlink()

    conn = get_connection()
    yields = load_yields(conn, start=args.start)
    conn.close()
    if yields.empty:
        raise SystemExit("No data in yields.db. Run collect.py first.")

    res = run_pca(yields)
    anomalies = detect_anomalies(res, threshold=args.threshold)

    if anomalies.empty:
        print(f"No anomalies above {args.threshold}-sigma in {len(res.scores)} days.")
        return

    print(f"{len(anomalies)} anomalies above {args.threshold}-sigma in this window.")
    alert_new(anomalies)


if __name__ == "__main__":
    main()
