const {JSDOM}=require('jsdom');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..');
const dom=new JSDOM(fs.readFileSync(path.join(root,'web/index.html'),'utf8'),{url:'https://kira.test/',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window,errors=[],calls=[];
w.addEventListener('error',e=>errors.push(e.message));
w.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
w.HTMLElement.prototype.scrollTo=()=>{};
let recognizer,started=0,stopped=0;
w.SpeechRecognition=class {
  constructor(){recognizer=this;}
  start(){started++;}
  stop(){stopped++;if(this.onend)this.onend();}
  abort(){stopped++;}
};
let projects=[],tasks=[],cards=[],documents=[],effects=0;
const respond=(data={},status=200)=>({status,ok:status>=200&&status<300,json:async()=>data});
w.fetch=async(url,opts={})=>{
  calls.push({url,opts});const route=new URL(url,'https://kira.test').pathname;const body=opts.body?JSON.parse(opts.body):{};
  if(route==='/api/auth/me')return respond({name:'Brice'});
  if(route==='/api/status')return respond({providers:[],budget:{used:0,paid_used:0,limit:1000},database:'postgres',version:'test',veille:{},devices:{},jobs:{running:{},last:{}}});
  if(route==='/api/conversations')return respond({conversations:[]});
  if(route==='/api/briefing')return respond({greeting:'Bonjour',suggestions:[],veille:{pending:0,top:[]}});
  if(route==='/api/projects'){
    if(opts.method==='POST'){projects.push({id:'p1',title:body.title,goal:body.goal,context:'',private:1,version:1});return respond(projects[0]);}
    return respond({projects});
  }
  if(route==='/api/projects/p1'){
    if(opts.method==='PATCH'){
      if(body.version!==projects[0].version)return respond({error:'Conflit entre appareils'},409);
      Object.assign(projects[0],body,{version:body.version+1});return respond(projects[0]);
    }
    return respond({project:{...projects[0]},tasks:tasks.map(t=>({...t})),cards:cards.map(c=>({...c})),documents:documents.map(d=>({...d}))});
  }
  if(route==='/api/projects/p1/tasks'){const t={id:'t1',title:body.title,state:'todo',result:'',version:1};tasks.push(t);return respond(t);}
  if(route==='/api/project-tasks/t1'){Object.assign(tasks[0],body,{version:body.version+1});return respond(tasks[0]);}
  if(route==='/api/projects/p1/cards'){const c={id:'c1',...body,due_at:0,correct:0,attempts:0,version:1};cards.push(c);return respond(c);}
  if(route==='/api/cards/c1/review'){effects++;Object.assign(cards[0],{due_at:Date.now()/1000+86400,correct:1,attempts:1,version:2});return respond(cards[0]);}
  if(route==='/api/projects/p1/documents'){assert.equal(Buffer.from(body.data,'base64').toString(),'Preuve Orion');documents.push({id:'d1',name:body.name,state:'ready',pages:1});return respond(documents[0]);}
  if(route==='/api/documents/d1/pages/1')return respond({name:'notes.txt',location:'section',number:1,content:'Preuve Orion'});
  if(route==='/api/jobs')return respond({items:[],running:{},last:{}});
  if(route==='/api/actions/pending')return respond({actions:[]});
  return respond({});
};
const tick=()=>new Promise(r=>setTimeout(r,2));
async function settle(){for(let i=0;i<10;i++)await tick();}
function button(label,scope='#project-detail'){return [...w.document.querySelectorAll(scope+' button')].find(b=>b.textContent===label);}
(async()=>{
  w.eval(fs.readFileSync(path.join(root,'web/app.js'),'utf8'));await settle();
  w.document.querySelector('[data-view=projects]').click();await settle();
  w.document.querySelector('#project-title').value='Orion';w.document.querySelector('#project-goal').value='Un projet durable';
  w.document.querySelector('#project-create').dispatchEvent(new w.Event('submit',{cancelable:true}));await settle();
  assert.match(w.document.querySelector('#project-detail').textContent,/Orion/);
  assert.equal(projects[0].private,1);
  const goal=w.document.querySelector('[aria-label="Objectif du projet"]');goal.value='Nouvel objectif';button('Enregistrer').click();await settle();
  assert.equal(projects[0].version,2);assert.equal(projects[0].goal,'Nouvel objectif');
  const taskInput=w.document.querySelector('[placeholder="Prochaine tâche"]');taskInput.value='Lire le document';taskInput.closest('form').dispatchEvent(new w.Event('submit',{cancelable:true}));await settle();
  assert.equal(tasks[0].title,'Lire le document');
  const state=w.document.querySelector('[aria-label="État de Lire le document"]');state.value='done';w.document.querySelector('[aria-label="Résultat de Lire le document"]').value='Lecture vérifiée';button('Enregistrer la tâche').click();await settle();assert.equal(tasks[0].state,'done');
  w.document.querySelector('[placeholder=Question]').value='Nom du projet ?';w.document.querySelector('[placeholder="Réponse de référence"]').value='Orion';
  w.document.querySelector('[placeholder=Question]').closest('form').dispatchEvent(new w.Event('submit',{cancelable:true}));await settle();
  button('Je maîtrise').click();await settle();assert.equal(effects,1);assert(button('Je maîtrise').disabled);
  const input=w.document.querySelector('[aria-label="Document du projet"]');Object.defineProperty(input,'files',{value:[new w.File(['Preuve Orion'],'notes.txt')]});button('Importer').click();await settle();
  assert.equal(documents.length,1);button('Ouvrir le texte').click();await settle();assert.match(w.document.querySelector('.document-text').textContent,/Preuve Orion/);
  button('Discuter de ce projet').click();await settle();assert.equal(w.document.querySelector('#view-chat').hidden,false);assert.match(w.document.querySelector('#chat-project').textContent,/Orion/);
  const mic=w.document.querySelector('#voice-mic');mic.click();assert.equal(started,0,'Aucun envoi audio sans accord explicite');
  w.document.querySelector('#voice-consent').checked=true;w.document.querySelector('#voice-locale').value='en-US';mic.click();assert.equal(recognizer.lang,'en-US');assert.equal(started,1);
  recognizer.onresult({resultIndex:0,results:[Object.assign([{transcript:'Hello Orion'}],{isFinal:false})]});assert.match(w.document.querySelector('#input').value,/Hello Orion/);
  recognizer.onend();assert.equal(started,2);mic.click();assert(stopped>0);assert.match(w.document.querySelector('#input').value,/Hello Orion/);
  assert.equal(calls.filter(c=>c.url==='/api/chat').length,0,'La dictée ne soumet jamais automatiquement le texte');
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('Interface projets OK : création, versions, tâches, documents citables, révisions et dictée manuelle FR/EN.');
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>dom.window.close());
