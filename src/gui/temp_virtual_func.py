    def _get_virtual_live_data(self):
        """
        Fetch LIVE Betfair markets and create 'Virtual' database rows for today's runners.
        Links to historical data via GreyhoundName -> GreyhoundID matching.
        """
        print("[VIRTUAL] Fetching live data from Betfair API...")
        
        try:
            from src.integration.betfair_fetcher import BetfairOddsFetcher
            import pandas as pd
            import re
            
            fetcher = BetfairOddsFetcher()
            if not fetcher.login():
                print("[VIRTUAL] Betfair login failed.")
                return pd.DataFrame()
                
            # 1. Fetch ALL Greyhound Markets (Next 24h)
            markets = fetcher.get_greyhound_markets()
            if not markets:
                print("[VIRTUAL] No live markets found.")
                fetcher.logout()
                return pd.DataFrame()
                
            print(f"[VIRTUAL] Found {len(markets)} markets from Betfair.")
            
            # 2. Get DB Mapping (DogName -> GreyhoundID)
            # We need this to link virtual rows to historical form
            conn = sqlite3.connect(self.db_path)
            # Fetch all dogs (or cache this if slow?) - distinct names is better
            # Actually, querying the whole Greyhounds table is fast enough (50k rows?)
            dog_map_df = pd.read_sql_query("SELECT GreyhoundID, GreyhoundName FROM Greyhounds", conn)
            # Create dictionary: Name -> ID
            # Normalize names: UPPER, strip
            dog_map_df['NormName'] = dog_map_df['GreyhoundName'].str.upper().str.strip()
            # Handle duplicates (take max ID?)
            dog_map = dict(zip(dog_map_df['NormName'], dog_map_df['GreyhoundID']))
            
            conn.close()
            print(f"[VIRTUAL] Loaded {len(dog_map)} dogs for history linking.")
            
            virtual_rows = []
            
            # 3. Process Markets
            for m in markets:
                # Track Name
                raw_track = m.event.name.split(' (')[0].split(' - ')[0].upper()
                clean_track = raw_track.replace('THE ', '').replace('MT ', 'MOUNT ').strip()
                
                # Race Number
                race_num = 0
                r_match = re.search(r'R(\d+)', m.market_name)
                if r_match:
                    race_num = int(r_match.group(1))
                    
                # Time
                start_dt = m.market_start_time
                if start_dt.tzinfo is None:
                    # Assume UTC if naive (Betfair standard)
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                local_dt = start_dt.astimezone() # To Local
                time_str = local_dt.strftime('%H:%M')
                
                # Race ID (Virtual)
                virtual_race_id = -1000 - int(m.market_id.replace('1.', '')) % 100000 
                
                for r in m.runners:
                    dog_name = re.sub(r'^\d+\.\s*', '', r.runner_name).strip().upper()
                    
                    # LINKING: Find GreyhoundID
                    gid = dog_map.get(dog_name)
                    
                    if not gid:
                        # Skip unmatched? Or include for manual betting?
                        # User said "entry in todays races need to be linked"
                        # But if we skip, we can't bet on them.
                        # Let's include them with gid=None/Negative
                        gid = -1
                        status = "UNLINKED"
                    else:
                        status = "LINKED"
                        
                    # Box
                    box = 1
                    try:
                        if hasattr(r, 'metadata') and r.metadata and 'TRAP' in r.metadata:
                            box = int(r.metadata['TRAP'])
                        else:
                            name_match = re.search(r'^(\d+)\.', r.runner_name)
                            if name_match: box = int(name_match.group(1))
                    except: pass
                    
                    # Create Row compatible with df_db columns
                    # EntryID, RaceID, GreyhoundID, Box, Position, BSP, Price5Min, Weight, TrainerID, Split, FinishTime, Margin, Distance, Grade, TrackName, MeetingDate, Dog, DateWhelped, RaceTime, RaceNumber
                    row = {
                        'EntryID': -1 * int(r.selection_id), # Virtual ID
                        'RaceID': virtual_race_id,
                        'GreyhoundID': gid,
                        'Box': box,
                        'Position': None,
                        'BSP': None,
                        'Price5Min': None, # Will be filled by injector
                        'Weight': 30.0, # Dummy
                        'TrainerID': None,
                        'Split': None,
                        'FinishTime': None,
                        'Margin': None,
                        'Distance': '300m', # Dummy if unknown
                        'Grade': 'M', # Dummy
                        'TrackName': clean_track,
                        'MeetingDate': local_dt.strftime('%Y-%m-%d'),
                        'Dog': dog_name,
                        'DateWhelped': None, 
                        'RaceTime': time_str,
                        'RaceNumber': race_num,
                        'LinkStatus': status, # For debug/UI
                        '_is_virtual': True
                    }
                    virtual_rows.append(row)
                    
            fetcher.logout()
            
            if not virtual_rows:
                print("[VIRTUAL] No runners extracted.")
                return pd.DataFrame()
                
            df_v = pd.DataFrame(virtual_rows)
            print(f"[VIRTUAL] Created {len(df_v)} virtual rows. Linked: {len(df_v[df_v['LinkStatus']=='LINKED'])}")
            
            return df_v
            
        except Exception as e:
            print(f"[VIRTUAL] Error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
