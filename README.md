# World-Cup-Predictor
Use the machine learning model developed from my personal projects and apply that to the world cup. Plug in my exact python predictor into Claude and ask to transition that into working for the World Cup. 


How it works

Power rating layer (your Quality Score analog): margin-weighted Elo over ~150 years of international results, with K scaled by match importance (World Cup 60, qualifiers/continental finals 50, friendlies 20) and a home-advantage term. USA, Mexico, and Canada get a partial host bump.
ML match layer: a multinomial logistic regression predicting win/draw/loss from Elo differential + 10-match rolling form, plus two Poisson regressions predicting goals scored — trained on 23,244 internationals since 2002. The Poisson layer matters because group-stage tiebreakers run on goal difference, and the classifier handles knockouts (draws split by relative strength to approximate extra time/pens).
Simulation layer: 20,000 Monte Carlo tournaments through the actual 2026 bracket — all 12 groups, the eight best third-place teams routed to their FIFA-constrained R32 slots via backtracking assignment, then R32 → Final.


What I used: the martj42 international results dataset on GitHub — every international since 1872, updated daily ; just a CSV pull
For live scores/fixtures: football-data.org has a free tier (10 calls/min) that covers the World Cup, and API-Football gives 100 free requests/day
For lines to compare against: The Odds API
