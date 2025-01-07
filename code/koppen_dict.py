# Köppen Description Dictionary
    koppen_descriptions = {
        'Dfc': 'Subarctic or boreal (taiga), cool summer',
        'Cfb': 'Warm-temperate, fully humid, warm summer',
        'Csa': 'Mediterranean, dry hot summer',
        'Cfa': 'Humid subtropical, no dry season, hot summer',
        'Dfb': 'Warm-summer humid continental, no dry season, warm summer',
        'Dfa': 'Hot-summer humid continental, no dry season, hot summer',
        'Dwc': 'Subarctic or boreal (taiga), dry winter, cool summer',
        'ET': 'Tundra, extremely cold, warmest month below 10°C',
        'Am': 'Tropical monsoon, short dry season',
        'Cwa': 'Humid subtropical, dry winter, hot summer',
        'Aw': 'Tropical savanna, dry winter',
        'Dfd': 'Subarctic or boreal (taiga), dry winter, extremely cold winter',
        'Af': 'Tropical rainforest, no dry season',
        'Cwc': 'Subarctic or boreal (taiga), dry winter, cool summer',
        'Dwa': 'Hot-summer humid continental, dry winter',
        'Bsh': 'Semi-arid (steppe), hot and dry'
    }

    # Map the descriptions to the DataFrame
    df_grouped['KOPPEN_Description'] = df_grouped['KOPPEN'].map(koppen_descriptions)
