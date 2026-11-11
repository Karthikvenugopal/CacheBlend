"""Read-only source audit and separately versioned agent annotations. No inference.

Only stdlib is imported: no tokenizer/model initialization, GPU context or network.
"""
import copy,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
ORIGINAL=ROOT/'experiments/cacheblend_overnight/frozen.json'
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/cacheblend_overnight/review_v2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n')

# Locations are source context indices (zero-based), with exact literal spans.
# Larger quotes explicitly anchor names and pronouns omitted from v1 excerpts.
SPECS={
 'musique:022':{
  'spans':[(0,'dimuthu bandara abayakoon ( born september 6, 1971 ) is a sri lankan politician, belonging to the janatha vimukthi peramuna.'),(5,'known as rohana wijeweera, was a sri lankan marxist politician, revolutionary and the founding leader of the janatha vimukthi peramuna.')],
  'steps':[{'evidence':[0],'claim':'Dimuthu Bandara Abayakoon belongs to Janatha Vimukthi Peramuna.'},{'evidence':[1],'claim':'Rohana Wijeweera is identified as the founding leader of that same named party.'},{'depends_on':[0,1],'operation':'Join on the explicit party name; return its founding leader.'}],
  'answer_from_text':'rohana wijeweera','notes':['The same window mentions his UPFA candidacy; party membership used here is the explicitly stated belonging-to relation, not an inferred coalition equivalence.']},
 'musique:027':{
  'spans':[(3,'the rabat ajax football ground is the training ground of maltese football team rabat ajax f. c. and is located in the town of mtarfa.'),(6,'malcolm licari ( born 18 april 1978 in pieta, malta ) is a retired professional footballer playing for pieta hotspurs, marsaxlokk, floriana, balzan and rabat ajax in the maltese premier league, where he plays as a striker, and also occasionally as a defender.')],
  'steps':[{'evidence':[0],'claim':'The named ground is used by Rabat Ajax F.C.'},{'evidence':[1],'claim':'The supplied account associates Rabat Ajax with the Maltese Premier League in its description of Licari.'},{'depends_on':[0,1],'operation':'Join the team names and return the league stated in the supplied account.'}],
  'answer_from_text':'maltese premier league','notes':['Text mixes retired/present-tense biographical language. This supports a source-conditioned historical association, not an independently checked current league affiliation. No present-day sports fact was looked up.']},
 'musique:030':{
  'spans':[(0,'in the shadow of greatness is a 2012 book written by 33 members of the united states naval academy class of 2002 and published by the united states naval institute.'),(0,'the united states naval institute ( usni ), based in annapolis, maryland, is a private, non - profit, professional military association')],
  'steps':[{'evidence':[0],'claim':'The book is published by the United States Naval Institute.'},{'evidence':[1],'claim':'That organization is based in Annapolis, Maryland.'},{'depends_on':[0,1],'operation':'Join on the organization name; interpret based-in as the location requested by the headquarters question.'}],
  'answer_from_text':'annapolis, maryland','notes':['Both hops occur in one intact retained window. The second window is additional context, not a missing supporting hop. Headquarters is a linguistic reading of based-in; no external headquarters record is assumed.']},
 'musique:033':{
  'spans':[(0,'snappy tomato pizza is a pizza chain that started in fort mitchell, kentucky, and has over 60 locations nationwide.'),(4,'fort mitchell is a home rule - class city in kenton county, kentucky, in the united states.')],
  'steps':[{'evidence':[0],'claim':'Snappy Tomato Pizza started in Fort Mitchell, Kentucky.'},{'evidence':[1],'claim':'Fort Mitchell is in Kenton County, Kentucky.'},{'depends_on':[0,1],'operation':'Join on the full city/state identity and return the county.'}],
  'answer_from_text':'kenton county','notes':['The question asks where the chain formed, so the stated started-in location is used rather than its separately stated headquarters in Burlington.']},
 'musique:039':{
  'spans':[(8,'the contemporary liberal party generally advocates economic liberalism'),(8,"strong opposition to socialism and communism in australia and abroad was one of its founding principles. the party's founder and longest - serving leader robert menzies envisaged that australia's middle class would form its main constituency."),(7,'politically, robert menzies and the liberal party of australia dominated much of the immediate post war era'),(7,"lyons'death in 1939 saw robert menzies assume the prime ministership on the eve of war.")],
  'steps':[{'evidence':[0,1,2],'claim':'The local party discourse, Australia context, named Australian Liberal Party and Robert Menzies supply the antecedent for the party-founder claim.'},{'evidence':[3],'claim':'The supplied account explicitly says Lyons death in 1939 led to Menzies assuming the prime ministership.'},{'depends_on':[0,1],'operation':'Join on Robert Menzies and return the stated event, reading become Prime Minister as his initial accession described here.'}],
  'answer_from_text':"lyons'death in 1939",'notes':['The abbreviated v1 quote omitted party identity. New contextual quotes supply that link from retained texts, not from the gold answer. The account also discusses later elections; this answer is conditional on the question referring to the initial accession.']},
 'wikimqa:025':{
  'spans':[(0,'guy arvely dolsin ( born january 15, 1957 in toamasina )'),(3,'altug celikbilek ( turkish pronunciation : [ ˈaɫtuː ˈtʃelicbilec ] ; born 7 september 1996 )')],
  'steps':[{'evidence':[0],'claim':'Guy Arvely Dolsin was born in 1957.'},{'evidence':[1],'claim':'The Altug Celikbilek entry gives a birth year of 1996.'},{'depends_on':[0,1],'operation':'Compare explicit birth years: 1996 is later than 1957, so the second person is younger.'}],
  'answer_from_text':'altug celikbilek','notes':['The question uses Altuğ Çelikbilek and the source uses an unaccented spelling with a pronunciation. This is an explicit agent name-normalization assumption, not human validation. Gold spelling is preserved without adding scoring aliases; exact scoring can penalize a source-faithful unaccented output.']},
 'wikimqa:026':{
  'spans':[(0,'jugband blues " is a song by the english psychedelic rock band pink floyd, released on their second album, a saucerful of secrets, in 1968.'),(1,'joel the lump of coal " joel the lump of coal " is a song by las vegas - based rock band the killers featuring late night talk show host jimmy kimmel. it was released on december 1, 2014.')],
  'steps':[{'evidence':[0],'claim':'Jugband Blues was released on the named album in 1968.'},{'evidence':[1],'claim':'The song named Joel the Lump of Coal is the antecedent of it in the December 1, 2014 release sentence.'},{'depends_on':[0,1],'operation':'Compare the provided release years: 1968 precedes 2014.'}],
  'answer_from_text':'jugband blues','notes':['V1 quoted only it was released... for the second song. The v2 span includes its title and subject sentence, closing the annotation gap without changing input text.']},
 'wikimqa:032':{
  'spans':[(9,'" fantasy " is a song by the english singer - songwriter george michael.'),(0,'michael died of heart disease on christmas day in 2016, at his home in goring - on - thames, oxfordshire. early life george michael was born georgios kyriacos panayiotou')],
  'steps':[{'evidence':[0],'claim':'Fantasy is performed by George Michael.'},{'evidence':[1],'claim':'The death sentence gives Michael home in Goring-on-Thames, Oxfordshire, followed by the same subject full-name early-life identification.'},{'depends_on':[0,1],'operation':'Resolve the local Michael subject to George Michael and return the stated place of death.'}],
  'answer_from_text':'goring - on - thames, oxfordshire','notes':['The paragraph transition and full name are now quoted explicitly. Home here is the stated death location, not an inferred birthplace or residence from an external biography.']},
 'wikimqa:035':{
  'spans':[(1,'above rubies is a 1932 british comedy film directed by frank richardson'),(1,'the magic aster the magic aster ( [UNK] [UNK] 花 ; ma lan hua ) is a chinese animated film released june 19, 2009')],
  'steps':[{'evidence':[0],'claim':'Above Rubies is dated 1932.'},{'evidence':[1],'claim':'The Magic Aster is explicitly named and its release dated June 19, 2009.'},{'depends_on':[0,1],'operation':'Compare the given film year and release year: 1932 precedes 2009.'}],
  'answer_from_text':'above rubies','notes':['V1 omitted the second film name from its date quotation. The v2 quote supplies the title already present in the frozen window.']}
}
GETTY_SPANS=[(1,'john paul getty jr. sir paul getty ( ; born eugene paul getty ; 7 september 1932 – 17 april 2003 ), known widely as john paul getty jr., was a british philanthropist and book collector.'),(1,'he was the third of five sons born to j. paul getty ( 1892 – 1976 )'),(1,'the getty family\'s wealth was the result of the oil business founded by george franklin getty. one of his sons, mark getty, co - founded the visual media company getty images.')]

def locate(item,ci,quote):
 assert ci in item['retained_context_indices']
 text=item['raw']['ctxs'][ci]['text'];start=text.index(quote);end=start+len(quote)
 assert text[start:end]==quote
 # Anchor the full source window in the original decoded input; no retokenization.
 cursor=0;starts={}
 for selected in item['retained_context_indices']:
  body=item['raw']['ctxs'][selected]['text'];position=item['decoded_input'].find(body,cursor)
  assert position>=0,(item['id'],selected,'full window not preserved in input')
  starts[selected]=position;cursor=position+len(body)
 pos=starts[ci]+start
 assert item['decoded_input'][pos:pos+len(quote)]==quote
 return {'source_file':f"experiments/cacheblend/reference/inputs__{item['workload']}_s.json",'source_record_index':item['source_index'],'json_location':f"[{item['source_index']}].ctxs[{ci}].text",'source_context_index_zero_based':ci,'assembled_chunk_index_zero_based':1+item['retained_context_indices'].index(ci),'source_character_span_half_open':[start,end],'decoded_prompt_character_span_half_open':[pos,pos+len(quote)],'quote':quote,'literal_location_check_passed':True}

def main():
 OUT.mkdir(parents=True,exist_ok=True);original_hash=sha(ORIGINAL);old=json.loads(ORIGINAL.read_text())
 old_report=json.loads((ROOT/'artifacts/cacheblend_overnight/evidence-manifest.json').read_text())
 for name,digest in old_report['sha256'].items():assert sha(ROOT/name)==digest,name
 findings=[];items=[]
 for item in old['items']:
  key=item['id']
  if key=='wikimqa:031':
   finding={'id':key,'decision':'exclude_ambiguous','annotation_origin':'agent_created','official':False,'human_validated':False,'evidence':[locate(item,c,q) for c,q in GETTY_SPANS],
    'supported_partial_hops':['The John Paul Getty Jr heading anchors the third-of-five-sons relation to J. Paul Getty.'],
    'unresolved_hops':['The Mark Getty parent link is expressed only as one of his sons immediately after George Franklin Getty is named. The intended return to the article subject John Paul Getty Jr is plausible but not explicitly disambiguated in this flattened text.'],
    'scoring_label_limits':['The retained windows do not explicitly expand the grandfather J. Paul Getty to the gold label Jean Paul Getty. This is a scoring-alias issue, not an additional reasoning hop or an independent reason to exclude. Do not manufacture an alias from the gold label.'],
    'reason':'Conservative exclusion for unresolved parent-link coreference. This does not assert the intended biography is false. The gold-name spelling issue is recorded separately and is not an answerability criterion; the agent will not fill the parent link from gold or outside knowledge.',
    'derived_answer':None,'replacement':None}
  else:
   spec=SPECS[key];evidence=[locate(item,c,q) for c,q in spec['spans']]
   finding={'id':key,'decision':'retain_with_agent_annotations','annotation_origin':'agent_created','official':False,'human_validated':False,'evidence':evidence,'reasoning_chain':spec['steps'],'derived_answer_from_quotes':spec['answer_from_text'],'necessary_hops':'present under the explicitly stated local linguistic/identity assumptions','assumptions_and_scoring_limits':spec['notes']}
   new=copy.deepcopy(item)
   new['legacy_agent_created_annotations_v1']=new.pop('manual_annotations')
   new['agent_evidence_review_v2']=finding
   new['original_input_ids_sha256']=hashlib.sha256(json.dumps(item['input_ids']).encode()).hexdigest()
   assert new['input_ids']==item['input_ids'] and new['chunks']==item['chunks']
   items.append(new)
  findings.append(finding)
 version={k:copy.deepcopy(v) for k,v in old.items() if k not in ('items','candidates','label','created_utc','preflight_item','selection')}
 version.update({'version':2,'created_utc':datetime.now(timezone.utc).isoformat(),'label':'AGENT-CREATED EVIDENCE-CONTROLLED DIAGNOSTIC; NOT OFFICIAL OR HUMAN-VALIDATED','annotation_origin':'agent_created','human_validated':False,'official_support_labels':False,'execution_authorized':False,'status':'review_only_do_not_launch','parent_frozen_file':str(ORIGINAL.relative_to(ROOT)),'parent_frozen_sha256':original_hash,'selection':'Review only the original frozen ten; retain nine, exclude ambiguous Getty; no replacement, no new candidates, no inference, no success-based selection. All retained input IDs and prompts are unchanged.','preflight_item':max(items,key=lambda r:(r['tokens'],r['id']))['id'],'items':items,'original_candidate_decisions_not_reaudited':old['candidates'],'reviewed_original_item_decisions':findings})
 target=HERE/'frozen-v2.json';assert not target.exists(),'Do not overwrite an existing version'
 write(target,version)
 summary={'parent_frozen_sha256':original_hash,'version2_sha256':sha(target),'original_items':10,'retained':len(items),'excluded':['wikimqa:031'],'new_candidates':0,'replacements':0,'input_ids_changed_for_retained_items':False,'original_candidate_pool':40,'original_selected':10,'original_excluded_before_run':26,'original_not_qualified_after_cap':4,'ranking_provenance':'not available; only supplied context order is known','prior_annotation_status':'Agent-created quotations and offsets existed; explicit chains and subject anchors were incomplete; no official or human validation existed. User approval authorized creating annotations, not validation of their contents.','findings':findings,'original_evidence_manifest_verified_files':len(old_report['sha256']),'inference_started':False}
 write(OUT/'annotation-audit.json',summary)
 assert sha(ORIGINAL)==original_hash
 print(json.dumps({k:v for k,v in summary.items() if k!='findings'},indent=2))
if __name__=='__main__':main()
