import sys, unicodedata, re
D=sys.argv[1]
dev=str.maketrans('०१२३४५६७८९','0123456789')
W={}
for f,off in [('numbers/digit.tsv',0),('numbers/teens_and_ties.tsv',0)]:
    for line in open(f'{D}/{f}',encoding='utf8'):
        p=line.rstrip('\n').split('\t')
        if len(p)==2: W[unicodedata.normalize('NFC',p[1])]=int(p[0].translate(dev))
W['शून्य']=0; W['ज़ीरो']=0; W['जीरो']=0
EN={'वन':1,'टू':2,'थ्री':3,'फोर':4,'फ़ोर':4,'फाइव':5,'सिक्स':6,'सेवन':7,'एट':8,'नाइन':9}
MULT={'सौ':100,'हज़ार':1000,'हजार':1000,'लाख':10**5,'करोड़':10**7,'अरब':10**9}
FRAC={'साढ़े':0.5,'सवा':0.25,'पौने':-0.25}
SPEC={'डेढ़':1.5,'ढाई':2.5}
def N(s): return unicodedata.normalize('NFC',s)
MULT={N(k):v for k,v in MULT.items()}; FRAC={N(k):v for k,v in FRAC.items()}; SPEC={N(k):v for k,v in SPEC.items()}
def parse(text):
    toks=[N(t) for t in re.findall(r'\S+',text)]
    out=[];cur=None
    i=0
    # amount mode: accumulate; digit mode: sequences of single digits
    total=0; grp=0; frac=None; seen=False; digits=[]
    def flush():
        nonlocal total,grp,seen,frac
        if seen: out.append(('AMT',total+grp)); 
        total=0;grp=0;seen=False;frac=None
    for t in toks:
        if t in ('डबल',): digits.append('x2'); continue
        if (t in W and W[t]<10 and not seen) or t in EN:
            v=W.get(t,EN.get(t))
            if digits and digits[-1]=='x2': digits[-1]=str(v); digits.append(str(v))
            else: digits.append(str(v))
            continue
        if digits:
            if len(digits)>=3: out.append(('DIG',''.join(digits)))
            else:
                for d in digits: grp+=int(d); seen=True  # short: treat as amount start
            digits=[]
        if t in FRAC: frac=FRAC[t]; continue
        if t in SPEC: grp=SPEC[t]; seen=True; continue
        if t in W: grp+=W[t]; seen=True; continue
        if t in MULT and seen:
            m=MULT[t]
            if frac is not None: grp=grp+frac; frac=None
            if m==100: grp*=100
            else: total+=grp*m; grp=0
            continue
        flush()
    if digits: out.append(('DIG',''.join(digits)))
    flush(); return out
for s in ['पच्चीस हज़ार रुपये','ढाई लाख','साढ़े तीन करोड़','पौने दो लाख','सवा पाँच सौ','एक करोड़ बीस लाख पचास हज़ार',
          'नंबर है नौ आठ सात छह पाँच चार तीन दो एक शून्य','नाइन एट डबल सात फाइव','पाँच लाख नहीं नहीं पाँच करोड़','सत्तासी हज़ार उनहत्तर','निन्यानवे','अट्ठानवे लाख']:
    print(s,'->',parse(s))
