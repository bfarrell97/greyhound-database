"""
Analyze the unstored API fields for betting value.
Pull fresh data and check win rates by jumpCode, sex, grade changes, etc.
"""
import json
from collections import defaultdict
from topaz_api import TopazAPI
from config import TOPAZ_API_KEY

api = TopazAPI(TOPAZ_API_KEY)

print("=" * 80)
print("ANALYZING UNSTORED API FIELDS FOR BETTING VALUE")
print("=" * 80)

# Collect data from multiple months
all_runs = []
states = ['VIC', 'NSW', 'QLD']  # Major states

for state in states:
    for month in range(1, 4):  # Jan-Mar 2025
        try:
            runs = api.get_bulk_runs_by_month(state, 2025, month)
            all_runs.extend(runs)
            print(f"  {state} 2025-{month:02d}: {len(runs)} runs")
        except Exception as e:
            print(f"  {state} 2025-{month:02d}: Error - {e}")

print(f"\nTotal runs collected: {len(all_runs)}")

# Filter to valid runs (not scratched, has finish position)
valid_runs = [r for r in all_runs if not r.get('scratched') and r.get('place') and r.get('place') <= 8]
print(f"Valid finishers: {len(valid_runs)}")

# ============================================================
# 1. JUMPCODE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("1. JUMPCODE (Quick/Medium/Slow) WIN RATE ANALYSIS")
print("=" * 80)

jump_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'price_sum': 0})
for run in valid_runs:
    jump = run.get('jumpCode') or 'Unknown'
    price = run.get('startPrice') or 0
    won = run.get('place') == 1
    
    jump_stats[jump]['total'] += 1
    if won:
        jump_stats[jump]['wins'] += 1
        if price:
            jump_stats[jump]['price_sum'] += price

print(f"\n{'JumpCode':<15} {'Wins':>8} {'Total':>10} {'Win%':>10} {'Avg Win Price':>15}")
print("-" * 60)
for jump in sorted(jump_stats.keys()):
    s = jump_stats[jump]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    avg_price = (s['price_sum'] / s['wins']) if s['wins'] > 0 else 0
    print(f"{jump:<15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}% {avg_price:>14.2f}")

# ============================================================
# 2. JUMPCODE BY BOX
# ============================================================
print("\n" + "=" * 80)
print("2. JUMPCODE WIN RATE BY BOX (Quick starters from inside?)")
print("=" * 80)

jump_box_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))
for run in valid_runs:
    jump = run.get('jumpCode') or 'Unknown'
    box = run.get('boxNumber') or 0
    won = run.get('place') == 1
    
    if box in [1, 2, 7, 8] and jump in ['Quick', 'Medium', 'Slow']:  # Focus on extreme boxes
        jump_box_stats[jump][box]['total'] += 1
        if won:
            jump_box_stats[jump][box]['wins'] += 1

print(f"\n{'JumpCode':<10} {'Box 1 Win%':>12} {'Box 2 Win%':>12} {'Box 7 Win%':>12} {'Box 8 Win%':>12}")
print("-" * 60)
for jump in ['Quick', 'Medium', 'Slow']:
    box1 = jump_box_stats[jump][1]
    box2 = jump_box_stats[jump][2]
    box7 = jump_box_stats[jump][7]
    box8 = jump_box_stats[jump][8]
    
    b1_pct = (box1['wins'] / box1['total'] * 100) if box1['total'] > 0 else 0
    b2_pct = (box2['wins'] / box2['total'] * 100) if box2['total'] > 0 else 0
    b7_pct = (box7['wins'] / box7['total'] * 100) if box7['total'] > 0 else 0
    b8_pct = (box8['wins'] / box8['total'] * 100) if box8['total'] > 0 else 0
    
    print(f"{jump:<10} {b1_pct:>11.1f}% {b2_pct:>11.1f}% {b7_pct:>11.1f}% {b8_pct:>11.1f}%")

# ============================================================
# 3. SEX ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("3. SEX (Dog/Bitch) WIN RATE ANALYSIS")
print("=" * 80)

sex_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'price_sum': 0})
for run in valid_runs:
    sex = run.get('sex') or 'Unknown'
    price = run.get('startPrice') or 0
    won = run.get('place') == 1
    
    sex_stats[sex]['total'] += 1
    if won:
        sex_stats[sex]['wins'] += 1
        if price:
            sex_stats[sex]['price_sum'] += price

print(f"\n{'Sex':<15} {'Wins':>8} {'Total':>10} {'Win%':>10} {'Avg Win Price':>15}")
print("-" * 60)
for sex in sorted(sex_stats.keys()):
    s = sex_stats[sex]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    avg_price = (s['price_sum'] / s['wins']) if s['wins'] > 0 else 0
    print(f"{sex:<15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}% {avg_price:>14.2f}")

# ============================================================
# 4. GRADE CHANGE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("4. GRADE CHANGE (Dogs dropping/rising in class)")
print("=" * 80)

grade_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'price_sum': 0})
for run in valid_runs:
    incoming = run.get('incomingGrade') or ''
    outgoing = run.get('outgoingGrade') or ''
    graded_to = run.get('gradedTo') or 'Unknown'
    price = run.get('startPrice') or 0
    won = run.get('place') == 1
    
    grade_stats[graded_to]['total'] += 1
    if won:
        grade_stats[graded_to]['wins'] += 1
        if price:
            grade_stats[graded_to]['price_sum'] += price

print(f"\n{'GradedTo':<15} {'Wins':>8} {'Total':>10} {'Win%':>10} {'Avg Win Price':>15}")
print("-" * 60)
for grade in sorted(grade_stats.keys()):
    s = grade_stats[grade]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    avg_price = (s['price_sum'] / s['wins']) if s['wins'] > 0 else 0
    print(f"{grade:<15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}% {avg_price:>14.2f}")

# ============================================================
# 5. FIRST SPLIT POSITION ANALYSIS  
# ============================================================
print("\n" + "=" * 80)
print("5. FIRST SPLIT POSITION (Leaders at first mark)")
print("=" * 80)

split_pos_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
for run in valid_runs:
    split_pos = run.get('firstSplitPosition')
    won = run.get('place') == 1
    
    if split_pos and split_pos <= 8:
        split_pos_stats[split_pos]['total'] += 1
        if won:
            split_pos_stats[split_pos]['wins'] += 1

print(f"\n{'1st Split Pos':>15} {'Wins':>8} {'Total':>10} {'Win%':>10}")
print("-" * 45)
for pos in sorted(split_pos_stats.keys()):
    s = split_pos_stats[pos]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    print(f"{pos:>15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}%")

# ============================================================
# 6. AGE ANALYSIS (from dateWhelped)
# ============================================================
print("\n" + "=" * 80)
print("6. AGE ANALYSIS (Peak performance age?)")
print("=" * 80)

from datetime import datetime

age_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
for run in valid_runs:
    whelped = run.get('dateWhelped')
    meeting_date = run.get('meetingDate')
    won = run.get('place') == 1
    
    if whelped and meeting_date:
        try:
            whelped_date = datetime.fromisoformat(whelped.replace('Z', '+00:00'))
            race_date = datetime.fromisoformat(meeting_date.replace('Z', '+00:00'))
            age_months = (race_date - whelped_date).days // 30
            age_years = age_months // 12
            
            if 1 <= age_years <= 6:
                age_stats[age_years]['total'] += 1
                if won:
                    age_stats[age_years]['wins'] += 1
        except:
            pass

print(f"\n{'Age (years)':>12} {'Wins':>8} {'Total':>10} {'Win%':>10}")
print("-" * 45)
for age in sorted(age_stats.keys()):
    s = age_stats[age]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    print(f"{age:>12} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}%")

# ============================================================
# 7. EXPERIENCE ANALYSIS (totalFormCount)
# ============================================================
print("\n" + "=" * 80)
print("7. EXPERIENCE (Career starts)")
print("=" * 80)

exp_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
for run in valid_runs:
    form_count = run.get('totalFormCount') or 0
    won = run.get('place') == 1
    
    # Bucket into experience levels
    if form_count == 0:
        exp = 'Debut'
    elif form_count <= 5:
        exp = '1-5 starts'
    elif form_count <= 15:
        exp = '6-15 starts'
    elif form_count <= 30:
        exp = '16-30 starts'
    else:
        exp = '31+ starts'
    
    exp_stats[exp]['total'] += 1
    if won:
        exp_stats[exp]['wins'] += 1

print(f"\n{'Experience':>15} {'Wins':>8} {'Total':>10} {'Win%':>10}")
print("-" * 45)
for exp in ['Debut', '1-5 starts', '6-15 starts', '16-30 starts', '31+ starts']:
    s = exp_stats[exp]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    print(f"{exp:>15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}%")

# ============================================================
# 8. CAREER PRIZE MONEY (Class indicator)
# ============================================================
print("\n" + "=" * 80)
print("8. CAREER PRIZE MONEY (High earners in lower races?)")
print("=" * 80)

prize_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
for run in valid_runs:
    career = run.get('careerPrizeMoney') or 0
    won = run.get('place') == 1
    
    # Bucket by earnings
    if career < 1000:
        bucket = '<$1k'
    elif career < 5000:
        bucket = '$1k-$5k'
    elif career < 15000:
        bucket = '$5k-$15k'
    elif career < 30000:
        bucket = '$15k-$30k'
    else:
        bucket = '$30k+'
    
    prize_stats[bucket]['total'] += 1
    if won:
        prize_stats[bucket]['wins'] += 1

print(f"\n{'Career $':>15} {'Wins':>8} {'Total':>10} {'Win%':>10}")
print("-" * 45)
for bucket in ['<$1k', '$1k-$5k', '$5k-$15k', '$15k-$30k', '$30k+']:
    s = prize_stats[bucket]
    win_pct = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
    print(f"{bucket:>15} {s['wins']:>8} {s['total']:>10} {win_pct:>9.1f}%")

# ============================================================
# 9. COMBINED: Quick starters from Box 1-2 with high earnings
# ============================================================
print("\n" + "=" * 80)
print("9. COMBINED ANGLE: Quick start + Box 1-2 + High earner")
print("=" * 80)

combo_stats = {'wins': 0, 'total': 0, 'price_sum': 0, 'return': 0}
for run in valid_runs:
    jump = run.get('jumpCode')
    box = run.get('boxNumber') or 0
    career = run.get('careerPrizeMoney') or 0
    price = run.get('startPrice') or 0
    won = run.get('place') == 1
    
    # Target: Quick starters from Box 1-2 with $15k+ career earnings
    if jump == 'Quick' and box in [1, 2] and career >= 15000:
        combo_stats['total'] += 1
        if won:
            combo_stats['wins'] += 1
            if price:
                combo_stats['price_sum'] += price
                combo_stats['return'] += price
        else:
            combo_stats['return'] -= 1

win_pct = (combo_stats['wins'] / combo_stats['total'] * 100) if combo_stats['total'] > 0 else 0
roi = (combo_stats['return'] / combo_stats['total'] * 100) if combo_stats['total'] > 0 else 0
avg_price = (combo_stats['price_sum'] / combo_stats['wins']) if combo_stats['wins'] > 0 else 0

print(f"\nQuick start + Box 1-2 + $15k+ career:")
print(f"  Total bets: {combo_stats['total']}")
print(f"  Wins: {combo_stats['wins']}")
print(f"  Win rate: {win_pct:.1f}%")
print(f"  Avg winning price: ${avg_price:.2f}")
print(f"  ROI: {roi:+.1f}%")
