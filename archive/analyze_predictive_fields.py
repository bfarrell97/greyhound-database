"""
Deep analysis of new API fields for predictive value.
Looking for angles that could improve the betting model.
"""
import sqlite3
from collections import defaultdict

DB_PATH = "greyhound_racing.db"

def analyze():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("PREDICTIVE VALUE ANALYSIS OF NEW FIELDS")
    print("=" * 80)
    
    # ============================================================
    # 1. JUMPCODE WIN RATES & ROI
    # ============================================================
    print("\n" + "=" * 80)
    print("1. JUMPCODE (Quick/Medium/Slow) - WIN RATE & ROI")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            JumpCode,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE JumpCode IS NOT NULL 
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY JumpCode
        ORDER BY wins * 1.0 / total DESC
    """)
    
    print(f"\n{'JumpCode':<12} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 50)
    for row in cursor.fetchall():
        jump, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{jump:<12} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 2. JUMPCODE BY BOX - Looking for edge
    # ============================================================
    print("\n" + "=" * 80)
    print("2. JUMPCODE x BOX - WIN RATE & ROI")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            JumpCode,
            Box,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE JumpCode IS NOT NULL 
          AND Box BETWEEN 1 AND 8
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY JumpCode, Box
        ORDER BY JumpCode, Box
    """)
    
    results = defaultdict(dict)
    for row in cursor.fetchall():
        jump, box, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        results[jump][box] = {'total': total, 'wins': wins, 'win_pct': win_pct, 'roi': roi}
    
    print(f"\n{'JumpCode':<10} {'Box':>4} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 55)
    for jump in ['Quick', 'Medium', 'Slow']:
        for box in [1, 2, 7, 8]:  # Focus on edge boxes
            if box in results[jump]:
                r = results[jump][box]
                print(f"{jump:<10} {box:>4} {r['total']:>10,} {r['wins']:>8,} {r['win_pct']:>7.1f}% {r['roi']:>+9.1f}%")
        print()

    # ============================================================
    # 3. FIRST SPLIT POSITION - Who leads early wins?
    # ============================================================
    print("\n" + "=" * 80)
    print("3. FIRST SPLIT POSITION - Early leaders win rate")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            FirstSplitPosition,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE FirstSplitPosition BETWEEN 1 AND 8
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY FirstSplitPosition
        ORDER BY FirstSplitPosition
    """)
    
    print(f"\n{'1st Split Pos':>14} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 55)
    for row in cursor.fetchall():
        pos, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{pos:>14} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 4. SEX - Dogs vs Bitches
    # ============================================================
    print("\n" + "=" * 80)
    print("4. SEX (Dog vs Bitch) - WIN RATE & ROI")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            g.Sex,
            COUNT(*) as total,
            SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN ge.Position = 1 THEN ge.StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE g.Sex IS NOT NULL 
          AND ge.Position IS NOT NULL 
          AND ge.Position != 'DNF'
          AND ge.StartingPrice > 0
        GROUP BY g.Sex
    """)
    
    print(f"\n{'Sex':<12} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 50)
    for row in cursor.fetchall():
        sex, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{sex:<12} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 5. AGE ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("5. AGE (Years) - WIN RATE & ROI")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            CAST((julianday(rm.MeetingDate) - julianday(g.DateWhelped)) / 365 AS INT) as age_years,
            COUNT(*) as total,
            SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN ge.Position = 1 THEN ge.StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE g.DateWhelped IS NOT NULL 
          AND ge.Position IS NOT NULL 
          AND ge.Position != 'DNF'
          AND ge.StartingPrice > 0
        GROUP BY age_years
        HAVING age_years BETWEEN 1 AND 6
        ORDER BY age_years
    """)
    
    print(f"\n{'Age (yrs)':>10} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 50)
    for row in cursor.fetchall():
        age, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{age:>10} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 6. GRADE CHANGES - Class droppers
    # ============================================================
    print("\n" + "=" * 80)
    print("6. GRADE CHANGES (IncomingGrade vs OutgoingGrade)")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN IncomingGrade = OutgoingGrade THEN 'Same Grade'
                WHEN IncomingGrade < OutgoingGrade THEN 'Promoted'
                WHEN IncomingGrade > OutgoingGrade THEN 'Demoted'
                ELSE 'Unknown'
            END as grade_change,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE IncomingGrade IS NOT NULL 
          AND OutgoingGrade IS NOT NULL
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY grade_change
        ORDER BY wins * 1.0 / total DESC
    """)
    
    print(f"\n{'Grade Change':<15} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 55)
    for row in cursor.fetchall():
        change, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{change:<15} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 7. CAREER PRIZE MONEY BUCKETS
    # ============================================================
    print("\n" + "=" * 80)
    print("7. CAREER PRIZE MONEY - Experience/Class indicator")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN CareerPrizeMoney < 1000 THEN '< $1k'
                WHEN CareerPrizeMoney < 5000 THEN '$1k-$5k'
                WHEN CareerPrizeMoney < 15000 THEN '$5k-$15k'
                WHEN CareerPrizeMoney < 30000 THEN '$15k-$30k'
                WHEN CareerPrizeMoney < 50000 THEN '$30k-$50k'
                ELSE '$50k+'
            END as career_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE CareerPrizeMoney IS NOT NULL 
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY career_bucket
        ORDER BY MIN(CareerPrizeMoney)
    """)
    
    print(f"\n{'Career $':<15} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 55)
    for row in cursor.fetchall():
        bucket, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{bucket:<15} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 8. AVERAGE SPEED QUINTILES
    # ============================================================
    print("\n" + "=" * 80)
    print("8. AVERAGE SPEED - Fastest dogs win more?")
    print("=" * 80)
    
    cursor.execute("""
        WITH speed_ranked AS (
            SELECT 
                ge.*,
                NTILE(5) OVER (ORDER BY ge.AverageSpeed DESC) as speed_quintile
            FROM GreyhoundEntries ge
            WHERE ge.AverageSpeed IS NOT NULL 
              AND ge.Position IS NOT NULL 
              AND ge.Position != 'DNF'
              AND ge.StartingPrice > 0
        )
        SELECT 
            speed_quintile,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns,
            ROUND(AVG(AverageSpeed), 2) as avg_speed
        FROM speed_ranked
        GROUP BY speed_quintile
        ORDER BY speed_quintile
    """)
    
    print(f"\n{'Quintile':<10} {'Avg Speed':>10} {'Total':>10} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 60)
    for row in cursor.fetchall():
        q, total, wins, returns, avg_speed = row
        label = ['Fastest', 'Fast', 'Medium', 'Slow', 'Slowest'][q-1]
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{label:<10} {avg_speed:>10.2f} {total:>10,} {wins:>8,} {win_pct:>7.1f}% {roi:>+9.1f}%")

    # ============================================================
    # 9. SIRE PERFORMANCE - Top producing sires
    # ============================================================
    print("\n" + "=" * 80)
    print("9. TOP SIRES BY WIN RATE (min 500 runners)")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            g.SireName,
            COUNT(*) as total,
            SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN ge.Position = 1 THEN ge.StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE g.SireName IS NOT NULL 
          AND ge.Position IS NOT NULL 
          AND ge.Position != 'DNF'
          AND ge.StartingPrice > 0
        GROUP BY g.SireName
        HAVING total >= 500
        ORDER BY wins * 1.0 / total DESC
        LIMIT 15
    """)
    
    print(f"\n{'Sire':<25} {'Total':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 60)
    for row in cursor.fetchall():
        sire, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{sire[:24]:<25} {total:>8,} {wins:>6,} {win_pct:>6.1f}% {roi:>+8.1f}%")

    # ============================================================
    # 10. COMBINED ANGLES - Looking for profitable combos
    # ============================================================
    print("\n" + "=" * 80)
    print("10. COMBINED ANGLES - Hunting for +ROI")
    print("=" * 80)
    
    # Quick + Box 1-2 + Various price ranges
    print("\n--- Quick Starters from Box 1-2 by Price Range ---")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN StartingPrice < 2 THEN '$1.00-$1.99'
                WHEN StartingPrice < 3 THEN '$2.00-$2.99'
                WHEN StartingPrice < 5 THEN '$3.00-$4.99'
                WHEN StartingPrice < 10 THEN '$5.00-$9.99'
                ELSE '$10+'
            END as price_range,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE JumpCode = 'Quick'
          AND Box IN (1, 2)
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY price_range
        ORDER BY MIN(StartingPrice)
    """)
    
    print(f"\n{'Price Range':<15} {'Total':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 50)
    for row in cursor.fetchall():
        price, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{price:<15} {total:>8,} {wins:>6,} {win_pct:>6.1f}% {roi:>+8.1f}%")

    # Quick + Box 1-2 + High Career Earnings
    print("\n--- Quick + Box 1-2 + Career $15k+ by Price ---")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN StartingPrice < 2 THEN '$1.00-$1.99'
                WHEN StartingPrice < 3 THEN '$2.00-$2.99'
                WHEN StartingPrice < 5 THEN '$3.00-$4.99'
                WHEN StartingPrice < 10 THEN '$5.00-$9.99'
                ELSE '$10+'
            END as price_range,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE JumpCode = 'Quick'
          AND Box IN (1, 2)
          AND CareerPrizeMoney >= 15000
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY price_range
        ORDER BY MIN(StartingPrice)
    """)
    
    print(f"\n{'Price Range':<15} {'Total':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 50)
    for row in cursor.fetchall():
        price, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{price:<15} {total:>8,} {wins:>6,} {win_pct:>6.1f}% {roi:>+8.1f}%")

    # Young dogs (2yo) with Quick start
    print("\n--- Young Dogs (2yo) + Quick Start ---")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN ge.StartingPrice < 3 THEN '$1-$2.99'
                WHEN ge.StartingPrice < 5 THEN '$3-$4.99'
                ELSE '$5+'
            END as price_range,
            COUNT(*) as total,
            SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN ge.Position = 1 THEN ge.StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.JumpCode = 'Quick'
          AND CAST((julianday(rm.MeetingDate) - julianday(g.DateWhelped)) / 365 AS INT) = 2
          AND ge.Position IS NOT NULL 
          AND ge.Position != 'DNF'
          AND ge.StartingPrice > 0
        GROUP BY price_range
        ORDER BY MIN(ge.StartingPrice)
    """)
    
    print(f"\n{'Price Range':<15} {'Total':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 50)
    for row in cursor.fetchall():
        price, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{price:<15} {total:>8,} {wins:>6,} {win_pct:>6.1f}% {roi:>+8.1f}%")

    # First Split Leader that DIDN'T win - fade them next time?
    print("\n--- Dogs who LED at 1st Split (pos=1) ---")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN StartingPrice < 2 THEN '$1.00-$1.99'
                WHEN StartingPrice < 3 THEN '$2.00-$2.99'
                WHEN StartingPrice < 5 THEN '$3.00-$4.99'
                ELSE '$5+'
            END as price_range,
            COUNT(*) as total,
            SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN Position = 1 THEN StartingPrice ELSE 0 END) as returns
        FROM GreyhoundEntries
        WHERE FirstSplitPosition = 1
          AND Position IS NOT NULL 
          AND Position != 'DNF'
          AND StartingPrice > 0
        GROUP BY price_range
        ORDER BY MIN(StartingPrice)
    """)
    
    print(f"\n{'Price Range':<15} {'Total':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 50)
    for row in cursor.fetchall():
        price, total, wins, returns = row
        win_pct = wins / total * 100 if total > 0 else 0
        roi = (returns - total) / total * 100 if total > 0 else 0
        print(f"{price:<15} {total:>8,} {wins:>6,} {win_pct:>6.1f}% {roi:>+8.1f}%")

    conn.close()
    
    print("\n" + "=" * 80)
    print("SUMMARY: Key findings for model improvement")
    print("=" * 80)
    print("""
Look for patterns where:
1. Win rate is HIGHER than expected for the odds
2. ROI is positive or close to breakeven
3. Sample size is large enough (1000+ bets)

Fields most likely to be predictive:
- JumpCode (Quick starters have edge)
- FirstSplitPosition (leaders at 1st split convert well)
- Age (younger dogs ~2yo have higher win rates)
- CareerPrizeMoney (experienced winners perform)
- AverageSpeed (faster dogs win more)
    """)

if __name__ == "__main__":
    analyze()
