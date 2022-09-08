# search name file with
# csvcut -c2 2021.csv | fzf
from datetime import datetime
import numpy as np
import pandas as pd

df = pd.read_csv('./data/2021.csv', index_col=0)
# df.index = df.index.rename('Team')
df.set_index('Player', inplace=True)

# [ins] In [5]: df.groupby('Tm').Age.mean().sort_values().median()
# Out[5]: 26.191123188405797


# STAT WEIGHTS # TODO tune these somehow
# TODO softmax these
weight_G = 1
weight_GS = 1
weight_Tgt = 1
weight_Rec = 1
weight_PassingYds = 1/20
weight_PassingTD = 2
weight_PassingAtt = .5
weight_RushingYds = 1/10
weight_RushingTD = 2
weight_RushingAtt = .5
weight_ReceivingYds = 1/10
weight_ReceivingTD = 2
weight_FantasyPoints = 2
weight_Int = -1
weight_Fumbles = -1
weight_FumblesLost = -2

all_stats = [
      'Player' # ix
    , 'Tm'  # team - gb
    , 'Pos' # position - gb
    , 'Age'
    , 'G'  # games
    , 'GS' # games started
    , 'Tgt'# RB, TE, WR
    , 'Rec' # receptions - WR, TE
    , 'PassingYds'  # QB
    , 'PassingTD'   # QB
    , 'PassingAtt'  # QB
    , 'RushingYds'  # RB, QB
    , 'RushingTD'   # RB, QB
    , 'RushingAtt'  # RB, QB
    , 'Int' # QB
    , 'ReceivingYds'    #WR
    , 'ReceivingTD'     #WR
    , 'FantasyPoints'   #WR
    , 'Fumbles' # ALL
    , 'FumblesLost' # All
]

z_cols = [
      'G'
    , 'GS'
    , 'Tgt'
    , 'Rec'
    , 'PassingYds'
    , 'PassingTD'
    , 'PassingAtt'
    , 'RushingYds'
    , 'RushingTD'
    , 'RushingAtt'
    , 'Int'
    , 'ReceivingYds'
    , 'ReceivingTD'
    , 'FantasyPoints'
    , 'Fumbles'
    , 'FumblesLost'
]

qb_stats = {
      'G': weight_G
    , 'GS': weight_GS
    , 'FantasyPoints': weight_FantasyPoints
    , 'PassingYds': weight_PassingYds
    , 'PassingTD': weight_PassingTD
    , 'PassingAtt': weight_PassingAtt
    , 'RushingYds': weight_RushingYds
    , 'RushingTD': weight_RushingTD
    , 'RushingAtt': weight_RushingAtt
    , 'Int': weight_Int
    , 'Fumbles': weight_Fumbles
    , 'FumblesLost': weight_FumblesLost
}

te_stats = {
      'G': weight_G
    , 'GS': weight_GS
    , 'FantasyPoints': weight_FantasyPoints
    , 'Tgt': weight_Tgt
    , 'Rec': weight_Rec
    , 'ReceivingYds': weight_ReceivingYds
    , 'ReceivingTD': weight_ReceivingTD
    , 'FantasyPoints': weight_FantasyPoints
    , 'Fumbles': weight_Fumbles
    , 'FumblesLost': weight_FumblesLost
}

rb_stats = {
      'G': weight_G
    , 'GS': weight_GS
    , 'FantasyPoints': weight_FantasyPoints
    , 'Tgt': weight_Tgt
    , 'RushingYds': weight_RushingYds
    , 'RushingTD': weight_RushingTD
    , 'RushingAtt': weight_RushingAtt
    , 'Fumbles': weight_Fumbles
    , 'FumblesLost': weight_FumblesLost
}

wr_stats = {
      'G': weight_G
    , 'GS': weight_GS
    , 'FantasyPoints': weight_FantasyPoints
    , 'Tgt': weight_Tgt
    , 'Rec': weight_Rec
    , 'ReceivingYds': weight_ReceivingYds
    , 'ReceivingTD': weight_ReceivingTD
    , 'Fumbles': weight_Fumbles
    , 'FumblesLost': weight_FumblesLost
}

pos_to_stats = {
      'WR': wr_stats
    , 'QB': qb_stats
    , 'RB': qb_stats
    , 'TE': te_stats
}


def score_player(row)->float:
    global pos_to_stats
    player_stats = pos_to_stats[row.Pos]
    stats = np.array(row[player_stats.keys()], dtype=float)
    weights = np.array([e for e in player_stats.values()], dtype=float)
    return np.dot(stats, weights)


# MAIN
df[z_cols] = (df[z_cols] - df[z_cols].mean())/df[z_cols].std()
position_weights = df.groupby('Pos').mean()['FantasyPoints']
position_weights /= position_weights.sum()
df.iloc[:10].apply(score_player, axis=1)


# Fill Nulls
unacceptable_null_cols = ['Tm', 'Pos']
df.dropna(subset=unacceptable_null_cols, inplace=True)

# Too many null columns. This guy is all nulls
# df.drop(index='KeeSean Johnson', inplace=True)

# acceptable_null_columns
df.Fumbles.mask(df.Fumbles.isna(), 0, inplace=True)
df.FantasyPoints.mask(df.FantasyPoints.isna(), 0, inplace=True)

player_scores = df.apply(score_player, axis=1)
df['score'] = player_scores

# df[['Pos', 'score']].sort_values('score', ascending=False)[:20]
player_scores = df[['Pos', 'score']].sort_values('score', ascending=False)

prompt = """r: remove player
s: save draft checkpoint
l: load draft
n: next page
t: top page
f: filter pos
q: quit
"""
page=1
while True:
    print(player_scores[30*(page-1):30*page], '\n')
    print(prompt)
    i = input('enter command:')
    if i=='f':
        pos = input('Enter Position (QB, TE, RB, WR):')
        print(player_scores[player_scores.Pos==pos][:30])
        input('Press any key to continue')
    if i=='q': break
    if i=='n': page+=1; continue
    if i=='t': page=1; continue
    if i=='s': # save checkpoint
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        player_scores.to_csv(f'player_scores_checkpoint_{ts}.csv')
    if i=='l': # load checkpoint
        fp = input('Enter draft checkpoint filepath:')
        player_scores = pd.read_csv(fp, index_col='Player')
    if i=='r' or i=='x': # remove player
        del_player = input('remove player:')
        try: player_scores.drop(index=del_player, inplace=True)
        except KeyError: print(del_player, 'not found')
    page=1
