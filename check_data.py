from statsbombpy import sb

# Check available data
comps = sb.competitions()
print(f'Total competition-seasons: {len(comps)}\n')

# Count by competition
print('Available competitions:')
comp_counts = comps.groupby('competition_name')['season_name'].count().sort_values(ascending=False)
for comp, count in comp_counts.head(10).items():
    print(f'  {comp}: {count} seasons')

print('\n' + '='*60)

# Check La Liga specifically
print('\nLa Liga seasons available:')
la_liga = comps[comps['competition_name'] == 'La Liga']
print(la_liga[['season_name', 'season_id']])

# Get matches for one season
print('\n' + '='*60)
print('\nFetching La Liga 2020/21 matches...')
matches = sb.matches(competition_id=11, season_id=90)
print(f'Total matches: {len(matches)}')
print(f'\nExample match: {matches.iloc[0]["home_team"]} vs {matches.iloc[0]["away_team"]}')
print(f'Date: {matches.iloc[0]["match_date"]}')

# Estimate total matches across all free data
print('\n' + '='*60)
print('\nEstimating total available matches (this may take a moment)...')
total_estimate = len(comps) * 50  # rough estimate
print(f'Approximate: {total_estimate}+ matches across all competitions')
