// The fork's uncensored families (setup.py FAMILIES orca / mrad / rvn, docs/FORK.md) reach the Models page.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
const compiled=await build({stdin:{contents:"export * from './src/lib/native-catalog'; export * from './src/lib/native-names'; export * from './src/lib/native-fit';",resolveDir:process.cwd()},bundle:true,write:false,platform:'node',format:'esm'});
const {nativeCatalogModels,nativeShortName,VERSION_GROUPS}=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const family=(id,title,models)=>({family:id,title,about:'',experimental:false,images:true,sizes:models.map(model=>({model,id:`${id}-${model.toLowerCase()}`}))});
test('orca, mrad and rvn are Qwen3.8-Flash-Next versions with their own edition',()=>{
 const [flash]=nativeCatalogModels([family('qwen','Qwen3.8-Flash-Next',['Q2_0']),family('orca','Uncensored (OrcaRouter)',['IQ2_M','IQ3_XXS']),
  family('mrad','Uncensored (mradermacher)',['IQ3_S']),family('rvn','Uncensored (RVN)',['IQ3_XS'])]);
 assert.deepEqual(flash.sizes.map(s=>[s.sourceFamily,s.model,s.edition]),[['qwen','Q2_0',''],['orca','IQ2_M','Uncensored (OrcaRouter)'],
  ['orca','IQ3_XXS','Uncensored (OrcaRouter)'],['mrad','IQ3_S','Uncensored (mradermacher)'],['rvn','IQ3_XS','Uncensored (RVN)']]);
});
test('their short names and their group',()=>{
 assert.equal(nativeShortName('orca-iq3_xxs'),'OrcaRouter IQ3_XXS');
 assert.equal(nativeShortName('mrad-iq3_s'),'mradermacher IQ3_S');
 assert.equal(nativeShortName('rvn-iq3_xs'),'RVN IQ3_XS');
 assert.deepEqual(VERSION_GROUPS.find(g=>g.id==='uncensored').families,['orca','mrad','rvn']);
});
