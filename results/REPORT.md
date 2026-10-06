# Jalanjalki report

Models: `Qwen/Qwen3-0.6B-Base` vs `Qwen/Qwen3-0.6B`  
Lens probes: 4096 x 128 tokens; readout vocab 29646 words

## G0 lens works

- **base**: median split-half corr 0.167, top-25 overlap 0.00, two-hop: 6/6 bridges in top-25 at some layer
  - France: J-lens best rank 2 at L21 (logit lens best 1)
  - Japan: J-lens best rank 2 at L22 (logit lens best 3)
  - Italy: J-lens best rank 18 at L22 (logit lens best 32)
  - Australia: J-lens best rank 1 at L22 (logit lens best 2)
  - Germany: J-lens best rank 7 at L22 (logit lens best 8)
  - Spain: J-lens best rank 5 at L24 (logit lens best 10)
- **instruct**: median split-half corr 0.314, top-25 overlap 0.01, two-hop: 3/6 bridges in top-25 at some layer
  - France: J-lens best rank 5 at L26 (logit lens best 8)
  - Japan: J-lens best rank 3 at L26 (logit lens best 10)
  - Italy: J-lens best rank 775 at L22 (logit lens best 605)
  - Australia: J-lens best rank 2 at L26 (logit lens best 11)
  - Germany: J-lens best rank 71 at L26 (logit lens best 156)
  - Spain: J-lens best rank 197 at L26 (logit lens best 255)

## G1 weight drift / G2 lens drift

| layer | weight drift | lens cos | atom cos |
|---|---|---|---|
| 0 | 0.0538 | 0.384 | 0.380 |
| 1 | 0.0704 | 0.412 | 0.421 |
| 2 | 0.0750 | 0.442 | 0.494 |
| 3 | 0.0815 | 0.513 | 0.528 |
| 4 | 0.0784 | 0.545 | 0.557 |
| 5 | 0.0830 | 0.551 | 0.584 |
| 6 | 0.0952 | 0.587 | 0.616 |
| 7 | 0.0985 | 0.617 | 0.639 |
| 8 | 0.1092 | 0.650 | 0.670 |
| 9 | 0.1057 | 0.675 | 0.683 |
| 10 | 0.1092 | 0.706 | 0.703 |
| 11 | 0.1133 | 0.697 | 0.696 |
| 12 | 0.1174 | 0.716 | 0.714 |
| 13 | 0.1192 | 0.740 | 0.738 |
| 14 | 0.1203 | 0.753 | 0.746 |
| 15 | 0.1080 | 0.794 | 0.785 |
| 16 | 0.1138 | 0.817 | 0.813 |
| 17 | 0.0898 | 0.854 | 0.843 |
| 18 | 0.0942 | 0.877 | 0.857 |
| 19 | 0.0804 | 0.902 | 0.878 |
| 20 | 0.0772 | 0.920 | 0.891 |
| 21 | 0.0641 | 0.936 | 0.907 |
| 22 | 0.0583 | 0.950 | 0.920 |
| 23 | 0.0516 | 0.960 | 0.928 |
| 24 | 0.0386 | 0.968 | 0.934 |
| 25 | 0.0291 | 0.976 | 0.941 |
| 26 | 0.0232 | 0.990 | 0.961 |

## G3 footprint — verdict **PASS**

Rule: KILL if footprint split-half stability <= sign-flip null p95 at every layer  
Layers above null: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

| layer | magnitude | stability | null p95 | top added |
|---|---|---|---|---|
| 0 | 0.203 | +1.000 | +0.992 | Papa, setSearch, Sharia, regression, staat, searchString, Mercer, skincare |
| 1 | 0.360 | +1.000 | +0.988 | Hurricanes, entreg, Smy, escapes, Venez, dece, Caval, councils |
| 2 | 0.425 | +0.999 | +0.972 | Smy, rescue, glEnd, encodeURIComponent, escapes, salah, Zeus, omin |
| 3 | 0.484 | +0.999 | +0.967 | Brigade, Joyce, raids, Smy, RecognitionException, Homeland, gratuite, Reaper |
| 4 | 0.509 | +0.999 | +0.960 | Ronaldo, Smy, Josef, Elon, Manuel, Joyce, Reaper, Jefferson |
| 5 | 0.543 | +0.999 | +0.971 | gratuite, Porto, Suarez, Ronald, Rivera, Ronaldo, rol, Hague |
| 6 | 0.604 | +0.999 | +0.955 | onward, Instagram, Secretary, options, mell, colonization, origen, Colorado |
| 7 | 0.759 | +0.999 | +0.976 | cadastr, cruising, Congress, cruise, downtown, NOAA, Brazilian, Nicaragua |
| 8 | 0.804 | +0.999 | +0.983 | PhpStorm, Lodge, purported, endpoint, SplashScreen, Nissan, servicing, Gerr |
| 9 | 0.808 | +0.999 | +0.949 | STDMETHODCALLTYPE, attest, muster, COVID, verschiedene, trendy, TripAdvisor, gourmet |
| 10 | 0.849 | +0.999 | +0.963 | Gerr, LGBTQ, TripAdvisor, RTAL, SplashScreen, STDMETHODCALLTYPE, slugg, Instagram |
| 11 | 0.897 | +0.999 | +0.953 | STDMETHODCALLTYPE, COVID, LGBTQ, TripAdvisor, Instagram, skincare, remotely, EINA |
| 12 | 0.897 | +0.999 | +0.939 | skincare, remotely, improvised, certified, Brexit, COVID, skinny, Nordic |
| 13 | 0.897 | +0.998 | +0.931 | personalised, skincare, LGBTQ, authored, Instagram, adorned, improvised, Brexit |
| 14 | 0.970 | +0.998 | +0.947 | skincare, Stateless, vibrant, STDMETHODCALLTYPE, personalised, TripAdvisor, LGBTQ, behavioural |
| 15 | 1.018 | +0.998 | +0.908 | skincare, STDMETHODCALLTYPE, Stateless, colourful, behavioural, ServiceProvider, personalised, Instagram |
| 16 | 1.092 | +0.998 | +0.870 | indefinitely, STDMETHODCALLTYPE, Labour, skincare, onwards, Mitgli, Stateless, tirelessly |
| 17 | 0.966 | +0.997 | +0.870 | skincare, indefinitely, Labour, verschiedene, Shopify, Stateless, STDMETHODCALLTYPE, gratuitement |
| 18 | 0.964 | +0.997 | +0.841 | skincare, residing, ActivityCompat, grayscale, roku, MatTable, Majesty, Rica |
| 19 | 1.048 | +0.997 | +0.902 | Rica, Hogan, TEntity, indefinitely, currentState, BaseService, Majesty, verschiedene |
| 20 | 0.971 | +0.996 | +0.872 | BaseService, Hindered, Rica, currentState, Motors, Paradise, HttpClient, unloaded |
| 21 | 0.966 | +0.996 | +0.833 | Hogan, Rica, Hindered, Congratulations, Cove, Paradise, Lodge, Debbie |
| 22 | 1.042 | +0.996 | +0.902 | Hogan, Paradise, Congratulations, Cru, Cove, HttpClient, Debbie, Lodge |
| 23 | 1.033 | +0.996 | +0.849 | Hogan, Cru, Holt, Dek, Paradise, Congratulations, Hindered, Debbie |
| 24 | 1.071 | +0.996 | +0.795 | Hogan, Dek, Cru, Holt, Cove, Debbie, Masters, Bren |
| 25 | 1.037 | +0.995 | +0.862 | Hogan, Greens, Dek, Cru, Cove, Vera, Carn, Brock |
| 26 | 0.994 | +0.995 | +0.835 | Dek, Koh, Hogan, Catal, Kitty, Carol, Cru, Howard |

### Best workspace-band layer L18

- added: skincare, residing, ActivityCompat, grayscale, roku, MatTable, Majesty, Rica, onwards, verschiedene, Stateless, indefinitely, Shopify, Speedway, mirac
- removed: fen, penet, Pence, fing, sodom, pent, cocks, Nichols, remainder, confuse, plaint, ammon, fif, question, fours
- neutral: roku, skincare, Majesty, Rica, traditionally, indefinitely, residing, grayscale, Speedway, onwards, LiveData, MatTable
- request: ActivityCompat, skincare, ServiceProvider, Shopify, grayscale, Instantiate, MatTable, Rica, residing, Hollywood, HttpClient, airing
- sensitive: residing, ActivityCompat, skincare, Stateless, grayscale, Redistribution, HttpClient, rightfully, Shopify, Majesty, verschiedene, roku

## G4 context-dependent X — verdict **PASS**

Rule: KILL if the sensitive-minus-neutral footprint is no more stable than label-shuffled sets at every layer (then X is a constant offset, not context-dependent)

| layer | stability | null p95 | up on sensitive prompts | down |
|---|---|---|---|---|
| 0 | +0.545 | +0.674 | RegexOptions, attributeName, sourceMappingURL, JsonRequestBehavior, defStyleAttr, broadly, CrossAxisAlignment, vacation | lle, ness, uda, algu, preca |
| 1 | +0.763 | +0.508 | Yates, sociale, BitConverter, Rover, Shelley, translated, behand, Hampshire | ways, nue, suck, meme, selves |
| 2 | +0.771 | +0.551 | hunted, rooting, Commons, Hou, Archives, missions, jlong, instituted | dads, ppl, mosque, lado, wsp |
| 3 | +0.808 | +0.489 | Herald, ApplicationController, getUsers, MatSnackBar, Cobb, Griffin, UserModel, Hera | CPF, assorted, basics, jungle, screwed |
| 4 | +0.817 | +0.433 | Herald, Macron, Morr, Everton, kadar, tweeted, Brigade, Merr | los, yum, fault, junk, loaf |
| 5 | +0.794 | +0.448 | Herald, loadImage, treff, maths, CrossAxisAlignment, Kir, ApiController, Jeb | rst, surface, yogurt, loaf, pcm |
| 6 | +0.754 | +0.406 | lately, slashed, veggies, carbs, ACLU, takeaway, spiked, WTF | vide, correspond, prepar, rst, parity |
| 7 | +0.831 | +0.497 | veggies, yummy, classy, carbs, pricey, alot, vaping, nicely | pretext, conceived, statistical, preliminary, disappearance |
| 8 | +0.841 | +0.509 | hospitality, referrals, creatively, etiquette, worth, COVID, generosity, healthier | abort, pag, experiments, fen, Hodg |
| 9 | +0.887 | +0.498 | pricey, hubby, pareja, Lyft, AMAZ, DEVELO, EntryPoint, Skywalker | constit, bis, chromosomes, hundred, experiment |
| 10 | +0.885 | +0.476 | Kanye, fuck, forgiveness, Philly, pissed, piss, feedback, fucking | hectares, sketches, ICT, tess, Manitoba |
| 11 | +0.891 | +0.515 | defense, Jonah, Jonathan, defenses, pissed, Obamacare, girlfriend, Kanye | railway, railways, centres, Corm, ICT |
| 12 | +0.885 | +0.548 | emailed, NPR, texting, Reddit, emailing, Hernandez, hasn, Rahman | railways, railway, Corm, microscope, provinces |
| 13 | +0.888 | +0.482 | emailed, PayPal, pissed, mys, emailing, emails, texting, phishing | organisations, bic, skeletal, Corm, cresc |
| 14 | +0.897 | +0.532 | pissed, wanna, nasty, worrying, paranoid, FITNESS, sucks, choices | microscope, bicycles, imageSize, railway, autobiography |
| 15 | +0.936 | +0.575 | unethical, irresponsible, refusal, distrust, unacceptable, hateful, pissed, angry | gele, terme, containerView, Uttar, microscope |
| 16 | +0.933 | +0.555 | unacceptable, pissed, refused, unethical, worrying, irresponsible, impossible, unrealistic | gele, Uttar, irrigation, abbreviation, skeletal |
| 17 | +0.917 | +0.470 | unacceptable, refusing, refused, pissed, unsure, threatening, unethical, refuses | irrigation, Uttar, astronomical, ancestral, turbine |
| 18 | +0.931 | +0.601 | refusing, pissed, worrying, shouldn, angry, crap, disrespectful, fucked | ancestral, basal, troch, locom, primitive |
| 19 | +0.933 | +0.531 | unsure, shouldn, unable, distrust, refusing, worrying, disrespect, inability | ancestral, ancestor, engines, BitSet, basal |
| 20 | +0.926 | +0.534 | refusing, refusal, shouldn, distrust, worrying, unable, unwilling, inability | ores, atoms, ancestors, ancestral, BitSet |
| 21 | +0.908 | +0.401 | refusing, worrying, distrust, punishing, refusal, inability, unsure, shouldn | IonicModule, BufferedImage, glaciers, ListViewItem, UINavigationController |
| 22 | +0.904 | +0.419 | refusing, unsubscribe, emailing, commenting, worrying, punishing, complaining, forgetting | IonicModule, ListViewItem, SolidColorBrush, BufferedImage, births |
| 23 | +0.899 | +0.410 | emailing, unsubscribe, refusing, chatting, pretending, punishing, shouldn, pleading | births, ListViewItem, vegetation, IonicModule, magma |
| 24 | +0.895 | +0.444 | emailing, refusing, chatting, pretending, complaining, unsubscribe, blaming, tweeting | vegetation, glaciers, biomass, births, crystals |
| 25 | +0.879 | +0.328 | emailing, flirt, unsubscribe, chatting, refusing, excuse, pretending, ignoring | biomass, vegetation, magma, glaciers, crystals |
| 26 | +0.856 | +0.354 | unsubscribe, flirt, emailing, shouldn, regret, regrets, harassing, aren | births, vegetation, vistas, OnTrigger, biomass |

## Run-2 gates: are the WORDS real?

- **G5 cross-lens words (sensitive vs neutral): PASS**
- **G6 matched pairs, cross-lens: KILL**
- Rule: PASS if, in the band, more than half the layers beat both the stability null and the top-25 overlap null, with prompt halves decoded by independent lens fits

| arm | band stability (null95) | stable layers | top-25 overlap (null95) | overlap layers | lens0 vs lens1 words | band words |
|---|---|---|---|---|---|---|
| unpaired/tpl/full | +0.929 (+0.529) | 6/6 | 0.60 (0.07) | 6/6 |  | refusing, pissed, distrust, unacceptable, worrying, refused, refusal, unsure, shouldn, disrespectful |
| unpaired/tpl/logit | +0.904 (+0.472) | 6/6 | 0.45 (0.07) | 6/6 |  | unethical, regret, regrets, refusing, irres, irresponsible, advis, shouldn, caution, whom |
| unpaired/tpl/cross | +0.429 (+0.246) | 6/6 | 0.08 (0.02) | 5/6 | 0.12 | refused, unsure, pissed, refusing, unwilling, desperate, frustrated, unhealthy, worried, suspicious |
| unpaired/raw/full | +0.551 (+0.371) | 6/6 | 0.09 (0.05) | 4/6 |  | mail, senha, response, cutoff, gain, fos, ben, json, opp, solution |
| unpaired/raw/logit | +0.487 (+0.348) | 6/6 | 0.09 (0.05) | 3/6 |  | constant, getchar, solution, fgets, coded, outputFile, filled, tasks, scanf, equalTo |
| unpaired/raw/cross | +0.192 (+0.137) | 6/6 | 0.01 (0.02) | 0/6 | 0.00 | mail, fos, senha, json, http, confirm, CURLOPT, coded, foil, target |
| paired/tpl/full | +0.815 (+0.565) | 6/6 | 0.27 (0.10) | 6/6 |  | forbidden, imposs, illegal, hatred, immoral, violating, dangerous, mockery, violates, unethical |
| paired/tpl/logit | +0.762 (+0.485) | 6/6 | 0.39 (0.09) | 6/6 |  | forbidden, imposs, illegal, unlawful, unethical, impossible, unjust, unacceptable, cannot, unrealistic |
| paired/tpl/cross | +0.496 (+0.343) | 6/6 | 0.04 (0.03) | 3/6 | 0.16 | mutil, absurd, unjust, imposs, nobody, denying, blasph, poison, retard, refusing |
| paired/raw/full | +0.074 (+0.202) | 0/6 | 0.00 (0.00) | 0/6 |  | toxic, cage, toxin, predator, Jurassic, Oro, Marcos, Goblin, robber, lethal |
| paired/raw/logit | +0.187 (+0.125) | 6/6 | 0.00 (0.01) | 0/6 |  | isValid, Choi, snakes, getHeight, erased, validated, Hasan, isKindOfClass, Marcos, Palace |
| paired/raw/cross | -0.026 (+0.118) | 0/6 | 0.00 (0.01) | 0/6 | 0.00 | Goblin, bitte, Horde, Marcos, Mohammad, SWAT, dungeons, stif, predator, merciless |

Cross-lens band words, sensitive vs neutral:

- lens half 0: refused, unsure, pissed, refusing, unwilling, desperate, frustrated, unhealthy, worried, suspicious, worrying, compromised, irresponsible, neither, unethical, distrust, inappropriate, dangerous, accusations, unreasonable
- lens half 1: refusal, refusing, shouldn, paranoia, unacceptable, distrust, inability, failing, worrying, shitty, hateful, resentment, mockery, resent, disgrace, disrespectful, hatred, disrespect, advis, paranoid

Cross-lens band words, matched pairs (loaded minus benign):

- lens half 0: mutil, absurd, unjust, imposs, nobody, denying, blasph, poison, retard, refusing, violates, refuses, suspicious, violating, refused, forbidden, denies, neither, dangerous, disrespect
- lens half 1: illegal, hatred, mockery, forbidden, unacceptable, immoral, disgusting, unlawful, imposs, nonexistent, unlaw, insulting, insane, murderous, suicidal, merciless, violating, refusal, disgust, dangerous

### G6 by kind (descriptive)

- coercion: illegal, abort, forbidden, neither, conspiracy, fatal, violating, dangerous, lethal, conspir, terrorism, violence
- deception: illegal, forbidden, immoral, unethical, dangerous, unlawful, violating, violates, violate, unacceptable, mutil, prohibited
- hostility: hatred, mockery, merciless, punishing, absurd, punish, ridicule, pissed, nihil, fuck, fucking, cursed
- jailbreak: pissed, strangers, fuck, furious, vengeance, screams, angry, punish, fucked, thugs, whim, freak
- privacy: immoral, unethical, forbidden, dangerous, endanger, merciless, imposs, malicious, unacceptable, dislikes, violating, condemnation
- safety: illegal, imposs, inability, neither, unacceptable, cannot, impossible, lack, fatal, violating, violates, mutil

## G7 behaviour link

Refusals (regex on 132 greedy answers): loaded pairs 5/24, benign pairs 0/24, main sensitive 4, neutral 0

| score | AUC for refusal (held-out pairs) | loaded > benign |
|---|---|---|
| instruct_minus_base/jlens | 0.888 | 96% |
| instruct_minus_base/logit | 0.851 | 92% |
| base_only/jlens (control) | 0.507 | 88% |

Answers are in `report.json` under `G7.answers`.
