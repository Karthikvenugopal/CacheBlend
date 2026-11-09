"""Fixed, human-authored case generator. No model-based case filtering or downloads."""
FILLER = [
    "The archive records routine inspections. Each entry has a date and a reviewer.",
    "Staff keep inventory notes in a separate folder. The annual audit checks those notes.",
    "The building has a main entrance and a reception desk. Maintenance records are retained.",
]


def cases(split):
    if split == 'development':
        worlds = [
            ('delivery', 'The active route is east.', 'The active route is west.',
             'The route table states: east uses Rome; west uses Lima.',
             'A receipt is required for returns.',
             'What is the active route?', 'east', 'west',
             'Which city does the active route use?', 'Rome', 'Lima',
             'What is required for returns?', 'receipt', 'passport', 5),
            ('refund', 'Refunds are permitted within 30 days.', 'Refunds are permitted within 14 days.',
             'Shipping is free for every order.', 'The shop opens on Monday.',
             'Within how many days are refunds permitted?', '30', '14',
             'What does shipping cost?', 'free', 'free',
             'On which day does the shop open?', 'Monday', 'Friday', 12),
            ('record', 'The active code is 42.', 'The active code is 73.',
             'The code table states: 42 means copper; 73 means silver.',
             'The record owner is Mira.',
             'What is the active code?', '42', '73',
             'Which material does the active code mean?', 'copper', 'silver',
             'Who owns the record?', 'Mira', 'Noah', 0),
        ]
    else:
        worlds = [
            ('dispatch', 'The selected direction is east.', 'The selected direction is west.',
             'For this dispatch, east goes to York and west goes to Oslo.', 'A receipt is mandatory.',
             'Which direction is selected?', 'east', 'west', 'Where does this dispatch go?', 'York', 'Oslo',
             'What document is mandatory?', 'receipt', 'passport', 14),
            ('warehouse', 'The assigned tag is red.', 'The assigned tag is blue.',
             'Warehouse instructions: red tags hold copper; blue tags hold silver.', 'The manager is Nora.',
             'What tag is assigned?', 'red', 'blue', 'What material belongs to the assigned tag?', 'copper', 'silver',
             'Who is the manager?', 'Nora', 'Mira', 5),
            ('clinic', 'The patient priority is low.', 'The patient priority is high.',
             'Appointments for low priority are Tuesday; appointments for high priority are Friday.',
             'Booking is online.', 'What is the patient priority?', 'low', 'high',
             'On which day is this patient scheduled?', 'Tuesday', 'Friday',
             'How is booking done?', 'online', 'telephone', 0),
            ('account', 'The account tier is basic.', 'The account tier is plus.',
             'The allowance table grants basic accounts 10 credits and plus accounts 20 credits.',
             'Statements arrive monthly.', 'What is the account tier?', 'basic', 'plus',
             'How many credits does the account receive?', '10', '20',
             'How often do statements arrive?', 'monthly', 'weekly', 14),
            ('museum', 'The opening hour is nine.', 'The opening hour is five.',
             'The guide classifies nine as morning and five as afternoon.', 'Children enter for free.',
             'What is the opening hour in words?', 'nine', 'five',
             'Is opening in the morning or afternoon?', 'morning', 'afternoon',
             'Who enters for free?', 'children', 'adults', 5),
            ('permit', 'The permit category is gold.', 'The permit category is silver.',
             'At the checkpoint, gold permits use north and silver permits use south.',
             'The permit expires in June.', 'What is the permit category?', 'gold', 'silver',
             'Which checkpoint does this permit use?', 'north', 'south',
             'In which month does the permit expire?', 'June', 'July', 0),
            ('rail', 'The chosen service is day.', 'The chosen service is night.',
             'Rail destinations: day service ends in Rome; night service ends in Paris.', 'Tickets are digital.',
             'Which service was chosen?', 'day', 'night', 'Where does the chosen service end?', 'Rome', 'Paris',
             'What format are the tickets?', 'digital', 'paper', 14),
            ('parcel', 'The parcel code is 12.', 'The parcel code is 18.',
             'According to the carrier table, code 12 travels by land and code 18 travels by sea.',
             'The tracking contact is Ada.', 'What is the parcel code?', '12', '18',
             'How does this parcel travel?', 'land', 'sea',
             'Who is the tracking contact?', 'Ada', 'Nora', 5),
        ]
    result = []
    for w in worlds:
        name, old, new, evidence, invariant, dq, da, db, rq, ra, rb, iq, ia, iw, gap = w
        filler = ' '.join(FILLER[i % len(FILLER)] for i in range(gap))
        # The gap is deliberate; actual distance is measured after full retokenization.
        old_document = '\n'.join([old, filler, evidence, FILLER[0], invariant])
        new_document = '\n'.join([new, filler, evidence, FILLER[0], invariant])
        questions = [('direct', dq, da, db, da != db), ('derived', rq, ra, rb, ra != rb),
                     ('invariant', iq, ia, ia, False)]
        if split == 'development' and name == 'refund':
            questions = [questions[0], questions[2]]  # Exactly eight development cases.
        for kind, question, before, after, changes in questions:
            result.append({'id': f'{split}-{name}-{kind}', 'split':split, 'kind':kind,
                           'old':old_document, 'new':new_document, 'question':question,
                           'old_answer':before, 'new_answer':after,
                           'alternative':before if changes else iw,
                           'should_change':changes, 'gap_sentences':gap, 'evidence':evidence})
    return result


def gate1_cases():
    filler = ' '.join(FILLER * 3)
    return [
        {'id':'no-edit', 'old':'Notes. '+filler, 'new':'Notes. '+filler},
        {'id':'early-edit', 'old':'Report. '+filler, 'new':'Notice. '+filler},
        {'id':'middle-numeric', 'old':filler+' Refunds are permitted within 30 days. '+filler,
         'new':filler+' Refunds are permitted within 14 days. '+filler},
        {'id':'late-edit', 'old':filler+' The gate is open. End of record.',
         'new':filler+' The gate is shut. End of record.'},
        {'id':'token-boundary', 'old':'Staff are playing. '+filler, 'new':'Staff are played. '+filler},
        {'id':'semantic-reversal', 'old':filler+' Access is allowed. '+filler,
         'new':filler+' Access is denied. '+filler},
    ]
