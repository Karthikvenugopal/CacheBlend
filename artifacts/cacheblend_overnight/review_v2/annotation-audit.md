# Frozen-input and evidence audit — version 2

**No inference was restarted. All annotations below are agent-created, not official and not human-validated.** Script checks establish that quotations occur at the recorded literal locations; they do not establish semantic correctness or human approval.

The original freeze selected ten items from a fixed pool of forty: indices20–39 in each of the two cached datasets, disjoint from the old pilot indices0–19. It capped qualification at five per workload, selected from agent judgments about support, ambiguity and the token budget, and consulted no model success for these IDs. The failed preflight generated no answers for these IDs. Of the original forty candidates, ten were included, twenty-six rejected with agent-supplied evidence/rubric reasons and four later Wiki items not qualified after the cap. The latter four are not established evidence failures. This audit rechecks the frozen ten, not all previously rejected candidates.

Each input retained exactly two supplied text windows, in supplied order, with no new truncation and a1536-token prompt limit. Inputs were1204–1356 tokens. When only one window was annotated as required, another fixed window was retained as additional context; it is not guaranteed to be an irrelevant distractor. Unrelated material inside all retained windows was preserved. The cached schema has no official support annotations, passage identifiers or retrieval scores/ranks; actual ranked-retrieval provenance is unknown. Original articles cannot be reconstructed or assumed complete from these windows.

Original annotations already contained two quotations and zero-based character offsets per included item, but did not give explicit hop-by-hop chains. Some excerpts left out their subjects or antecedents. The preparing script called them human evidence annotations; that was inaccurate. They were created by the agent. The user authorized their creation, not the factual validation of the individual annotations. Version2 corrects this provenance and supplies explicit subject anchors and chains.

**Decision:** retain nine items with explicitly stated source-conditioned reasoning; conservatively exclude `wikimqa:031` (Getty). No replacement or new candidate was introduced. Every retained prompt, input ID, gold label, chunk boundary, generation setting and selection ratio is unchanged. The original10-item freeze and failed-run files are untouched. The largest retained prompt is still Wiki25 at1356 tokens: this review does not evade the memory failure by choosing a shorter preflight.

The original freeze SHA-256 is `6cd0b3bf5d384372a29c377c5cdc2f0be69c8b42babf84c9409d450cf91fe519`. The [corrected review-only set](../../../experiments/cacheblend_overnight/review_v2/frozen-v2.json) has SHA-256 `68030f866f02f8a51f3860efaffe053a90fa886d871d066ab454ad87ea99bb4b`. It is explicitly marked `execution_authorized: false`. It is not connected to an active launcher. Original candidate decisions are retained as historical agent judgments, not newly validated labels.

Locations below use the source record index and zero-based `ctxs` index. Character spans are zero-based, half-open Unicode-character offsets into the original `text` field. Assembled chunk0 is the instruction prefix. Machine-readable records also include exact decoded-prompt character spans. No external sources or gold-derived missing relations were used. Original gold labels remain metadata and were not expanded into new aliases.

## musique:022 — retain

Evidence 0: `[22].ctxs[0].text`, chars `[253, 377)`, assembled chunk1.

> dimuthu bandara abayakoon ( born september 6, 1971 ) is a sri lankan politician, belonging to the janatha vimukthi peramuna.

Evidence 1: `[22].ctxs[5].text`, chars `[1194, 1329)`, assembled chunk2.

> known as rohana wijeweera, was a sri lankan marxist politician, revolutionary and the founding leader of the janatha vimukthi peramuna.

Hop 0: Dimuthu Bandara Abayakoon belongs to Janatha Vimukthi Peramuna. Evidence: [0].

Hop 1: Rohana Wijeweera is identified as the founding leader of that same named party. Evidence: [1].

Conclusion from hops [0, 1]: Join on the explicit party name; return its founding leader.

Text-supported answer: `rohana wijeweera`.

Assumption/limit: The same window mentions his UPFA candidacy; party membership used here is the explicitly stated belonging-to relation, not an inferred coalition equivalence.

## musique:027 — retain

Evidence 0: `[27].ctxs[3].text`, chars `[912, 1045)`, assembled chunk1.

> the rabat ajax football ground is the training ground of maltese football team rabat ajax f. c. and is located in the town of mtarfa.

Evidence 1: `[27].ctxs[6].text`, chars `[823, 1082)`, assembled chunk2.

> malcolm licari ( born 18 april 1978 in pieta, malta ) is a retired professional footballer playing for pieta hotspurs, marsaxlokk, floriana, balzan and rabat ajax in the maltese premier league, where he plays as a striker, and also occasionally as a defender.

Hop 0: The named ground is used by Rabat Ajax F.C. Evidence: [0].

Hop 1: The supplied account associates Rabat Ajax with the Maltese Premier League in its description of Licari. Evidence: [1].

Conclusion from hops [0, 1]: Join the team names and return the league stated in the supplied account.

Text-supported answer: `maltese premier league`.

Assumption/limit: Text mixes retired/present-tense biographical language. This supports a source-conditioned historical association, not an independently checked current league affiliation. No present-day sports fact was looked up.

## musique:030 — retain

Evidence 0: `[30].ctxs[0].text`, chars `[453, 617)`, assembled chunk1.

> in the shadow of greatness is a 2012 book written by 33 members of the united states naval academy class of 2002 and published by the united states naval institute.

Evidence 1: `[30].ctxs[0].text`, chars `[112, 247)`, assembled chunk1.

> the united states naval institute ( usni ), based in annapolis, maryland, is a private, non - profit, professional military association

Hop 0: The book is published by the United States Naval Institute. Evidence: [0].

Hop 1: That organization is based in Annapolis, Maryland. Evidence: [1].

Conclusion from hops [0, 1]: Join on the organization name; interpret based-in as the location requested by the headquarters question.

Text-supported answer: `annapolis, maryland`.

Assumption/limit: Both hops occur in one intact retained window. The second window is additional context, not a missing supporting hop. Headquarters is a linguistic reading of based-in; no external headquarters record is assumed.

## musique:033 — retain

Evidence 0: `[33].ctxs[0].text`, chars `[730, 845)`, assembled chunk1.

> snappy tomato pizza is a pizza chain that started in fort mitchell, kentucky, and has over 60 locations nationwide.

Evidence 1: `[33].ctxs[4].text`, chars `[1536, 1627)`, assembled chunk2.

> fort mitchell is a home rule - class city in kenton county, kentucky, in the united states.

Hop 0: Snappy Tomato Pizza started in Fort Mitchell, Kentucky. Evidence: [0].

Hop 1: Fort Mitchell is in Kenton County, Kentucky. Evidence: [1].

Conclusion from hops [0, 1]: Join on the full city/state identity and return the county.

Text-supported answer: `kenton county`.

Assumption/limit: The question asks where the chain formed, so the stated started-in location is used rather than its separately stated headquarters in Burlington.

## musique:039 — retain

Evidence 0: `[39].ctxs[8].text`, chars `[549, 619)`, assembled chunk2.

> the contemporary liberal party generally advocates economic liberalism

Evidence 1: `[39].ctxs[8].text`, chars `[852, 1094)`, assembled chunk2.

> strong opposition to socialism and communism in australia and abroad was one of its founding principles. the party's founder and longest - serving leader robert menzies envisaged that australia's middle class would form its main constituency.

Evidence 2: `[39].ctxs[7].text`, chars `[1352, 1459)`, assembled chunk1.

> politically, robert menzies and the liberal party of australia dominated much of the immediate post war era

Evidence 3: `[39].ctxs[7].text`, chars `[1008, 1095)`, assembled chunk1.

> lyons'death in 1939 saw robert menzies assume the prime ministership on the eve of war.

Hop 0: The local party discourse, Australia context, named Australian Liberal Party and Robert Menzies supply the antecedent for the party-founder claim. Evidence: [0, 1, 2].

Hop 1: The supplied account explicitly says Lyons death in 1939 led to Menzies assuming the prime ministership. Evidence: [3].

Conclusion from hops [0, 1]: Join on Robert Menzies and return the stated event, reading become Prime Minister as his initial accession described here.

Text-supported answer: `lyons'death in 1939`.

Assumption/limit: The abbreviated v1 quote omitted party identity. New contextual quotes supply that link from retained texts, not from the gold answer. The account also discusses later elections; this answer is conditional on the question referring to the initial accession.

## wikimqa:025 — retain

Evidence 0: `[25].ctxs[0].text`, chars `[307, 363)`, assembled chunk1.

> guy arvely dolsin ( born january 15, 1957 in toamasina )

Evidence 1: `[25].ctxs[3].text`, chars `[1893, 1985)`, assembled chunk2.

> altug celikbilek ( turkish pronunciation : [ ˈaɫtuː ˈtʃelicbilec ] ; born 7 september 1996 )

Hop 0: Guy Arvely Dolsin was born in 1957. Evidence: [0].

Hop 1: The Altug Celikbilek entry gives a birth year of 1996. Evidence: [1].

Conclusion from hops [0, 1]: Compare explicit birth years: 1996 is later than 1957, so the second person is younger.

Text-supported answer: `altug celikbilek`.

Assumption/limit: The question uses Altuğ Çelikbilek and the source uses an unaccented spelling with a pronunciation. This is an explicit agent name-normalization assumption, not human validation. Gold spelling is preserved without adding scoring aliases; exact scoring can penalize a source-faithful unaccented output.

## wikimqa:026 — retain

Evidence 0: `[26].ctxs[0].text`, chars `[253, 392)`, assembled chunk1.

> jugband blues " is a song by the english psychedelic rock band pink floyd, released on their second album, a saucerful of secrets, in 1968.

Evidence 1: `[26].ctxs[1].text`, chars `[361, 548)`, assembled chunk2.

> joel the lump of coal " joel the lump of coal " is a song by las vegas - based rock band the killers featuring late night talk show host jimmy kimmel. it was released on december 1, 2014.

Hop 0: Jugband Blues was released on the named album in 1968. Evidence: [0].

Hop 1: The song named Joel the Lump of Coal is the antecedent of it in the December 1, 2014 release sentence. Evidence: [1].

Conclusion from hops [0, 1]: Compare the provided release years: 1968 precedes 2014.

Text-supported answer: `jugband blues`.

Assumption/limit: V1 quoted only it was released... for the second song. The v2 span includes its title and subject sentence, closing the annotation gap without changing input text.

## wikimqa:031 — exclude

Evidence 0: `[31].ctxs[1].text`, chars `[813, 998)`, assembled chunk2.

> john paul getty jr. sir paul getty ( ; born eugene paul getty ; 7 september 1932 – 17 april 2003 ), known widely as john paul getty jr., was a british philanthropist and book collector.

Evidence 1: `[31].ctxs[1].text`, chars `[999, 1066)`, assembled chunk2.

> he was the third of five sons born to j. paul getty ( 1892 – 1976 )

Evidence 2: `[31].ctxs[1].text`, chars `[1171, 1346)`, assembled chunk2.

> the getty family's wealth was the result of the oil business founded by george franklin getty. one of his sons, mark getty, co - founded the visual media company getty images.

Conservative exclusion for unresolved parent-link coreference. This does not assert the intended biography is false. The gold-name spelling issue is recorded separately and is not an answerability criterion; the agent will not fill the parent link from gold or outside knowledge.

Present: The John Paul Getty Jr heading anchors the third-of-five-sons relation to J. Paul Getty.

Unresolved: The Mark Getty parent link is expressed only as one of his sons immediately after George Franklin Getty is named. The intended return to the article subject John Paul Getty Jr is plausible but not explicitly disambiguated in this flattened text.

Scoring issue, not a missing reasoning hop or independent exclusion criterion: the retained text does not explicitly expand J. Paul Getty to the gold label Jean Paul Getty. Do not manufacture an alias from the gold label.

## wikimqa:032 — retain

Evidence 0: `[32].ctxs[9].text`, chars `[1766, 1838)`, assembled chunk2.

> " fantasy " is a song by the english singer - songwriter george michael.

Evidence 1: `[32].ctxs[0].text`, chars `[320, 489)`, assembled chunk1.

> michael died of heart disease on christmas day in 2016, at his home in goring - on - thames, oxfordshire. early life george michael was born georgios kyriacos panayiotou

Hop 0: Fantasy is performed by George Michael. Evidence: [0].

Hop 1: The death sentence gives Michael home in Goring-on-Thames, Oxfordshire, followed by the same subject full-name early-life identification. Evidence: [1].

Conclusion from hops [0, 1]: Resolve the local Michael subject to George Michael and return the stated place of death.

Text-supported answer: `goring - on - thames, oxfordshire`.

Assumption/limit: The paragraph transition and full name are now quoted explicitly. Home here is the stated death location, not an inferred birthplace or residence from an external biography.

## wikimqa:035 — retain

Evidence 0: `[35].ctxs[1].text`, chars `[192, 263)`, assembled chunk2.

> above rubies is a 1932 british comedy film directed by frank richardson

Evidence 1: `[35].ctxs[1].text`, chars `[1248, 1360)`, assembled chunk2.

> the magic aster the magic aster ( [UNK] [UNK] 花 ; ma lan hua ) is a chinese animated film released june 19, 2009

Hop 0: Above Rubies is dated 1932. Evidence: [0].

Hop 1: The Magic Aster is explicitly named and its release dated June 19, 2009. Evidence: [1].

Conclusion from hops [0, 1]: Compare the given film year and release year: 1932 precedes 2009.

Text-supported answer: `above rubies`.

Assumption/limit: V1 omitted the second film name from its date quotation. The v2 quote supplies the title already present in the frozen window.

No new quality estimate follows from this annotation review. V2 is an agent-audited evidence-controlled diagnostic, with the stated local linguistic assumptions, not an official/human-validated answerability benchmark.

A preliminary metadata draft is preserved separately. The final revision distinguishes the Getty parent-link ambiguity (the exclusion reason) from its gold-name alias issue (a scoring limitation). This refinement changed no item IDs or token IDs and involved no inference.
