"""
test_chip.py — 测试筹码指标对"前期没涨就跌"回调的区分效果
"""
import sys, numpy as np, random
sys.path.insert(0, '.')
from src.core.storage.database import Database
from src.server.plugin_mgr import indicators
from plugins.pullback_scanner.backend.scanner import (
    _find_decline_events, _compute_trough_features, _compute_recovery, assess_pullback,
)

db = Database()
random.seed(777); np.random.seed(777)
codes = random.sample(db.get_active_stock_codes(), 300)

all_events = []
print(f'扫描 300 只股票, 提取筹码指标...')

for ci, code in enumerate(codes):
    rows = db.fetchall(
        'SELECT trade_date, open, high, low, close, volume, turn FROM daily_kline WHERE code=? ORDER BY trade_date', (code,))
    if len(rows) < 120: continue
    dates = [r['trade_date'] for r in rows]
    rc = np.array([r['close'] for r in rows], float)
    rh = np.array([r['high'] for r in rows], float)
    rl = np.array([r['low'] for r in rows], float)
    ro = np.array([r['open'] for r in rows], float)
    rv = np.array([r['volume'] or 0 for r in rows], float)
    rturn = np.array([r['turn'] or 0 for r in rows], float)
    afs = db.fetchall('SELECT trade_date, adj_factor, fore_factor FROM adjust_factor WHERE code=? ORDER BY trade_date', (code,))
    adj = indicators.forward_adjust(rc, rh, rl, ro, dates, afs)
    close, high, low = adj['close'], adj['high'], adj['low']

    events = _find_decline_events(close, high, low, dates, min_decline=3, max_decline=50)
    name_row = db.fetchone('SELECT name FROM stock_basic WHERE code=?', (code,))
    name = name_row['name'] if name_row else ''

    for e in events:
        pk = e['peak_idx']; tr = e['trough_idx']
        g60 = round((close[pk] - close[max(0, pk-60)]) / close[max(0, pk-60)] * 100, 1)
        if g60 >= 0: continue

        features = _compute_trough_features(close, high, low, pk, tr)
        features['decline_pct'] = e['decline_pct']
        prob = assess_pullback(features)
        if prob < 80: continue

        recovery = _compute_recovery(dates, close, tr, e['peak_price'])
        n = len(close); ti = tr

        # ==== 筹码指标 ====
        pb_len = max(tr - pk, 1)

        # 1. 缩量比: 回调期均量 / 回调前10日均量
        pre_vol = float(np.mean(rv[max(0, pk-10):pk])) if pk >= 10 else float(np.mean(rv[:pk])) if pk > 0 else 0
        pb_vol = float(np.mean(rv[pk:tr+1]))
        vol_ratio = round(pb_vol / pre_vol, 2) if pre_vol > 0 else None

        # 2. 回调期累计换手率
        cum_turn = round(float(np.sum(rturn[pk:tr+1])), 1)

        # 3. 低点在60日价格区间的位置 (0=区间底, 100=区间顶)
        rng_hi = float(np.max(high[max(0, tr-60):tr+1]))
        rng_lo = float(np.min(low[max(0, tr-60):tr+1]))
        rng_s = rng_hi - rng_lo
        price_pos_60d = round((close[tr] - rng_lo) / rng_s * 100, 1) if rng_s > 0 else None

        # 4. 回调期日均换手 / 60日均换手
        avg_turn_60d = float(np.mean(rturn[max(0, tr-60):tr+1]))
        daily_turn = cum_turn / pb_len
        turn_ratio = round(daily_turn / avg_turn_60d, 2) if avg_turn_60d > 0 else None

        # Forward
        fwd_60 = round((close[ti+60]-close[ti])/close[ti]*100, 1) if ti+60 < n else None

        all_events.append({
            'code': code, 'name': name,
            'trough_date': e['trough_date'],
            'decline_pct': e['decline_pct'],
            'gain_60d': g60, 'rsi': features['rsi'],
            'probability': prob,
            'vol_ratio': vol_ratio, 'cum_turn': cum_turn,
            'price_pos_60d': price_pos_60d, 'turn_ratio': turn_ratio,
            'recovery_days': recovery['recovery_days'] if recovery else None,
            'r60d': fwd_60,
        })

    if (ci+1) % 50 == 0:
        print(f'  进度: {ci+1}/300, 已找到 {len(all_events)} 个')

print(f'\n找到 {len(all_events)} 个事件\n')

# ===== Analysis =====
def analyze_group(label, group):
    if not group: return
    rec = [r for r in group if r['recovery_days'] is not None]
    rate = len(rec)/len(group)*100
    med = np.median([r['recovery_days'] for r in rec]) if rec else 0
    r60s = [r['r60d'] for r in group if r['r60d'] is not None]
    r60a = np.mean(r60s) if r60s else 0
    pos60 = len([v for v in r60s if v>0])/max(len(r60s),1)*100
    print(f'  {label:<18}: {len(group):>5}次  恢复率{rate:.0f}%  中位{med:.0f}d  60日{r60a:+.1f}%  胜率{pos60:.0f}%')

# Baseline
rec_all = [r for r in all_events if r['recovery_days']]
print('--- 基线 (仅 前期没涨+概率>=80%) ---')
print(f'  总数: {len(all_events)}  恢复率: {len(rec_all)/max(len(all_events),1)*100:.0f}%  中位: {np.median([r["recovery_days"] for r in rec_all]):.0f}d')

# 1. Vol ratio
print('\n--- 1. 缩量比 (回调期均量/回调前均量, <1=缩量) ---')
for lo, hi, label in [(0, 0.5, '极度缩量 <0.5'), (0.5, 0.8, '明显缩量 0.5-0.8'),
                       (0.8, 1.0, '轻度缩量 0.8-1.0'), (1.0, 1.5, '放量 1.0-1.5'), (1.5, 5, '大幅放量 >1.5')]:
    analyze_group(label, [r for r in all_events if r['vol_ratio'] is not None and lo <= r['vol_ratio'] < hi])

# 2. 60d position
print('\n--- 2. 低点在60日区间位置 (0=底, 100=顶) ---')
for lo, hi, label in [(0, 5, '极低位 0-5%'), (5, 15, '低位 5-15%'), (15, 30, '中低位 15-30%'),
                       (30, 50, '中位 30-50%'), (50, 100, '高位 >50%')]:
    analyze_group(label, [r for r in all_events if r['price_pos_60d'] is not None and lo <= r['price_pos_60d'] < hi])

# 3. Turn ratio
print('\n--- 3. 换手活跃度 (回调日均换手/60日均换手) ---')
for lo, hi, label in [(0, 0.5, '极不活跃 <0.5'), (0.5, 0.8, '不活跃 0.5-0.8'),
                       (0.8, 1.2, '正常 0.8-1.2'), (1.2, 2.0, '活跃 1.2-2.0'), (2.0, 99, '异常活跃 >2.0')]:
    analyze_group(label, [r for r in all_events if r['turn_ratio'] is not None and lo <= r['turn_ratio'] < hi])

# 4. Best combo: 缩量 + 低位
print('\n--- 4. ★ 最佳组合: 缩量(<0.8) + 60日低位(<15%) ---')
g_best = [r for r in all_events
          if r['vol_ratio'] is not None and r['vol_ratio'] < 0.8
          and r['price_pos_60d'] is not None and r['price_pos_60d'] < 15]
analyze_group('缩量+低位', g_best)
# Show examples
if g_best:
    print('  样本:')
    for r in sorted(g_best, key=lambda r: r['decline_pct'])[:10]:
        rec = f"{r['recovery_days']}d" if r['recovery_days'] else '未'
        print(f'    {r["code"]} {r["name"]:<6} {r["trough_date"]} 跌{r["decline_pct"]}% RSI{r["rsi"]:.0f} 缩量{r["vol_ratio"]:.1f} 位{r["price_pos_60d"]:.0f}% 恢复{rec}')

# 5. Worst combo: 放量 + 高位
print('\n--- 5. ★ 危险组合: 放量(>1.2) + 中高位(>30%) ---')
g_bad = [r for r in all_events
         if r['vol_ratio'] is not None and r['vol_ratio'] > 1.2
         and r['price_pos_60d'] is not None and r['price_pos_60d'] > 30]
analyze_group('放量+高位', g_bad)

# 6. Decline + chip combo overview
print('\n--- 6. 跌幅 + 缩量 交叉 ---')
for dl, dh, dlbl in [(3, 8, '跌3-8%'), (8, 12, '跌8-12%'), (12, 18, '跌12-18%'), (18, 50, '跌>18%')]:
    base = [r for r in all_events if dl <= r['decline_pct'] < dh]
    shrink = [r for r in base if r['vol_ratio'] is not None and r['vol_ratio'] < 0.8]
    if not base: continue
    rec_base = len([r for r in base if r['recovery_days']])/len(base)*100
    rec_shrink = len([r for r in shrink if r['recovery_days']])/max(len(shrink),1)*100
    print(f'  {dlbl}: 全部{rec_base:.0f}%({len(base)}次) → 加缩量{rec_shrink:.0f}%({len(shrink)}次)')
