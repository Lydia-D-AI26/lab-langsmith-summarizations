import tempfile,unittest
from pathlib import Path
from summary_lab import *

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.records=[{'original_id':'t1','page':1,'modality':'text','content':'Alpha revenue grows','summary':'Alpha revenue grows','dpi':None},
        {'original_id':'t2','page':2,'modality':'text','content':'Beta visual layout','summary':'Beta visual layout','dpi':None}]
    def test_join_persistence(self):
        index=SummaryIndex(self.records)
        self.assertEqual(index.retrieve('Alpha',1)[0]['content'],'Alpha revenue grows')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'index.json';index.save(p)
            self.assertEqual(SummaryIndex.load(p).retrieve('Alpha',1)[0]['page'],1)
    def test_caps_audit_sent_only(self):
        selected,sent=select_evidence(self.records,max_text=1,max_images=0)
        answer=PDFAnswer(answer='Claim',evidence_pages=[2],confidence='high',limitations=[],claims=[Claim(text='Claim',pages=[2])])
        audit=audit_citations(answer,sent)
        self.assertFalse(audit['provenance_valid'])
        self.assertEqual(audit['invalid_pages'],[2])
    def test_empty_citations(self):
        answer=PDFAnswer(answer='Unknown',evidence_pages=[],confidence='low',limitations=['no evidence'])
        audit=audit_citations(answer,[1])
        self.assertTrue(audit['provenance_valid']);self.assertFalse(audit['cited_anything'])
    def test_cache_invalidation(self):
        item=self.records[0]
        key=item_cache_key(item,'model','prompt')
        self.assertEqual(key,item_cache_key({**item,'page':8},'model','prompt'))
        self.assertNotEqual(key,item_cache_key({**item,'content':'Changed'},'model','prompt'))
        self.assertNotEqual(key,item_cache_key(item,'new-model','prompt'))
        self.assertNotEqual(key,item_cache_key(item,'model','new-prompt'))
    def test_routing(self):
        self.assertEqual(route_query(NUMERIC_QUERY)['route'],'numeric')
        self.assertEqual(route_query('Describe the cover')['route'],'visual')
        self.assertEqual(route_query('Summarize the thesis')['route'],'narrative')
    def test_local_honesty(self):
        result=answer_pdf(SummaryIndex(self.records),'Alpha',mode='local')
        self.assertIn('NOT RUN',result['answer'].answer)
        self.assertEqual(result['answer'].evidence_pages,[])
    def test_numeric_company_association(self):
        answer=PDFAnswer(answer='Values',evidence_pages=[5],confidence='high',limitations=[],numeric_rows=[NumericRow(company='MongoDB',ev_ntm=14.6,ntm_growth_percent=17)])
        self.assertTrue(numeric_check(answer)['MongoDB'])
        answer.numeric_rows[0].company='Cloudflare'
        self.assertFalse(any(numeric_check(answer).values()))
    def test_claim_page_union(self):
        answer=PDFAnswer(answer='Claim',evidence_pages=[1],confidence='high',limitations=[],claims=[Claim(text='Claim',pages=[2])])
        self.assertFalse(audit_citations(answer,[1,2])['claim_citations_consistent'])
if __name__=='__main__':unittest.main()
