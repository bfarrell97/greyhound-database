"""
Compare what data the Topaz API provides vs what we're storing.
Identify potentially useful data we're NOT capturing.
"""

print("=" * 80)
print("TOPAZ API DATA COMPARISON: Available vs Stored")
print("=" * 80)

# What the API provides (from bulk runs endpoint)
api_fields = {
    # CURRENTLY STORING (mapped to our columns)
    'dogId': 'GreyhoundID',
    'dogName': 'GreyhoundName (in Greyhounds table)',
    'boxNumber': 'Box',
    'weightInKg': 'Weight',
    'trainerId': 'TrainerID',
    'ownerId': 'OwnerID',
    'place': 'Position',
    'resultMargin': 'Margin',
    'pir': 'InRun (position in running)',
    'resultTime': 'FinishTime',
    'firstSplitTime': 'Split',
    'startPrice': 'StartingPrice',
    'last5': 'Form',
    'comment': 'Comment',
    'rating': 'Rating (but showing 0% populated)',
    
    # NOT STORING - POTENTIALLY USEFUL
    'jumpCode': 'NOT STORED - Quick/Medium/Slow start',
    'startPaceCode': 'NOT STORED - Early pace indicator',
    'runLineCode': 'NOT STORED - Normal/Wide/Checked run',
    'averageSpeed': 'NOT STORED - Average race speed',
    'firstSplitPosition': 'NOT STORED - Position at first mark',
    'secondSplitPosition': 'NOT STORED - Position at second mark',
    'secondSplitTime': 'NOT STORED - Second split time',
    'incomingGrade': 'NOT STORED - Grade before race',
    'outgoingGrade': 'NOT STORED - Grade after race',
    'gradedTo': 'NOT STORED - Grading result (Field/Up/Down)',
    'sireId': 'NOT STORED - Father ID',
    'sireName': 'NOT STORED - Father name',
    'damId': 'NOT STORED - Mother ID', 
    'damName': 'NOT STORED - Mother name',
    'dateWhelped': 'NOT STORED - Birth date (age)',
    'sex': 'NOT STORED - Dog/Bitch',
    'colourCode': 'NOT STORED - BK/BD/F etc',
    'careerPrizeMoney': 'NOT STORED - Total career earnings',
    'totalFormCount': 'NOT STORED - Number of career starts',
    'prizeMoney': 'NOT STORED - Prize money for this race',
    'resultMarginLengths': 'NOT STORED - Margin as lengths string',
    'bestTime': 'NOT STORED - FSH/NBT/time',
    'bestTimeTrackDistance': 'NOT STORED - Best at this track/distance',
    'trainerSuburb': 'NOT STORED - Trainer location',
    'trainerDistrict': 'NOT STORED - Trainer region',
    'ownerState': 'NOT STORED - Owner state',
    'isBestBet': 'NOT STORED - Official best bet flag',
    'isQuad': 'NOT STORED - In quaddie flag',
    'firstSecond': 'NOT STORED - Who ran 1st/2nd',
}

print("\n" + "=" * 80)
print("KEY UNSTORED FIELDS WITH BETTING POTENTIAL")
print("=" * 80)

valuable_fields = """
1. jumpCode (Quick/Medium/Slow)
   - Direct measure of box start quality
   - Could filter for Quick starters from inside boxes
   - Quick start + Box 1-2 might be valuable angle

2. averageSpeed (km/h)
   - Absolute speed measure (not relative to field)
   - Could identify class drops (fast dogs in slow fields)
   - Better than benchmark for cross-track comparison?

3. firstSplitPosition (position at first mark)
   - Shows actual early racing position
   - Different from InRun which is full race positions
   - Leaders at first split have higher win rates?

4. incomingGrade / outgoingGrade / gradedTo
   - Grade changes (UP/DOWN/Field)
   - Class droppers might be undervalued
   - Dogs graded DOWN after a race = improving form?

5. sireId/sireName + damId/damName
   - Breeding data for sire/dam performance patterns
   - Some sires produce better early speed dogs
   - Breeding angles widely used in horse racing

6. dateWhelped (birth date)
   - Age of dog at race time
   - Young dogs improving vs old dogs declining
   - Peak age performance window?

7. sex (Dog/Bitch)
   - Bitches may perform differently in heat cycles
   - Potential market inefficiency

8. careerPrizeMoney
   - Measure of career quality
   - High earners dropping in class = value?

9. totalFormCount (career starts)
   - Experience level
   - First start dogs vs veterans

10. bestTimeTrackDistance
    - Best performance at THIS track/distance combo
    - Track specialists might be undervalued
"""

print(valuable_fields)

print("\n" + "=" * 80)
print("RECOMMENDED SCHEMA ADDITIONS")
print("=" * 80)

schema_additions = """
ALTER TABLE GreyhoundEntries:
  ADD COLUMN JumpCode TEXT           -- 'Quick', 'Medium', 'Slow'
  ADD COLUMN AverageSpeed REAL       -- km/h average for run
  ADD COLUMN FirstSplitPosition INT  -- Position at first split
  ADD COLUMN SecondSplitTime REAL    -- Second split time
  ADD COLUMN SecondSplitPosition INT -- Position at second split
  ADD COLUMN IncomingGrade TEXT      -- Grade before race
  ADD COLUMN OutgoingGrade TEXT      -- Grade after race
  ADD COLUMN GradedTo TEXT           -- 'Field', 'Up', 'Down'
  ADD COLUMN PrizeMoney REAL         -- Prize for this race

ALTER TABLE Greyhounds:
  ADD COLUMN SireID INT              -- Father ID
  ADD COLUMN SireName TEXT           -- Father name
  ADD COLUMN DamID INT               -- Mother ID
  ADD COLUMN DamName TEXT            -- Mother name
  ADD COLUMN DateWhelped DATE        -- Birth date
  ADD COLUMN Sex TEXT                -- 'Dog' or 'Bitch'
  ADD COLUMN ColourCode TEXT         -- 'BK', 'BD', 'F', etc.
"""

print(schema_additions)

print("\n" + "=" * 80)
print("QUICK WIN: CHECK IF JUMPCODE IS VALUABLE")
print("=" * 80)

quick_analysis = """
The jumpCode field is immediately interesting because:
1. It's a categorical measure of box start quality
2. Could correlate strongly with Box 1-2 win rates
3. Would explain some of the "slow early split = higher win rate" anomaly
   (maybe those dogs have Medium/Slow jump but stronger runs?)

To test: Pull recent data with jumpCode, analyze win rates by:
  - Quick starters from Box 1 vs Box 8
  - Quick starters at short vs long distances
  - Jump code vs actual first split time
"""

print(quick_analysis)
