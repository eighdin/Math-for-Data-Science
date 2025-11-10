// Simple logistic regression trainer in plain JS
document.addEventListener('DOMContentLoaded', ()=>{
  const csvInput = document.getElementById('csvInput');
  const lrInput = document.getElementById('learningRate');
  const iterInput = document.getElementById('iterations');
  const trainBtn = document.getElementById('trainBtn');
  const outCsv = document.getElementById('outCsv');
  const logEl = document.getElementById('log');
  const downloadBtn = document.getElementById('downloadBtn');
  const initSelect = document.getElementById('initSelect');

  // Example dataset: two features, label (0/1). Simple separable set.
  csvInput.value = `2.7810836,2.550537003,0
1.465489372,2.362125076,0
3.396561688,4.400293529,0
1.38807019,1.850220317,0
3.06407232,3.005305973,0
7.627531214,2.759262235,1
5.332441248,2.088626775,1
6.922596716,1.77106367,1
8.675418651,-0.242068655,1
7.673756466,3.508563011,1`;

  function parseCSV(text){
    const rows = text.split(/\r?\n/).map(r=>r.trim()).filter(r=>r.length>0);
    const X = [];
    const y = [];
    for(const r of rows){
      const parts = r.split(',').map(s=>s.trim());
      if(parts.length<2) continue;
      const nums = parts.map(p=>Number(p));
      if(nums.some(n=>Number.isNaN(n))) continue;
      const label = nums[nums.length-1];
      y.push(label);
      const features = nums.slice(0, nums.length-1);
      X.push(features);
    }
    return {X,y};
  }

  function sigmoid(z){
    // clip for numeric stability
    if(z>30) return 1;
    if(z<-30) return 0;
    return 1/(1+Math.exp(-z));
  }

  function computeLossAndGrad(X,y,weights){
    const N = X.length;
    const D = weights.length;
    let loss = 0;
    const grad = new Array(D).fill(0);
    const eps = 1e-12;
    for(let i=0;i<N;i++){
      const xi = X[i]; // already includes bias as first element
      let z = 0;
      for(let j=0;j<D;j++) z += weights[j]*xi[j];
      const yhat = sigmoid(z);
      // cross-entropy
      loss += -( y[i]*Math.log(yhat+eps) + (1-y[i])*Math.log(1-yhat+eps) );
      const diff = (yhat - y[i]);
      for(let j=0;j<D;j++) grad[j] += diff * xi[j];
    }
    for(let j=0;j<D;j++) grad[j] /= N;
    loss /= N;
    return {loss,grad};
  }

  function prepareData(parsed){
    const X = parsed.X.map(row=>{
      // add bias term at index 0
      return [1,...row];
    });
    return {X,y:parsed.y};
  }

  function train(X,y,opts){
    const N = X.length;
    if(N===0) throw new Error('No data');
    const D = X[0].length;
    let weights = new Array(D).fill(0);
    if(opts.init==='random'){
      for(let j=0;j<D;j++) weights[j] = (Math.random()-0.5);
    }
    const history = [];
    for(let it=0; it<opts.iterations; it++){
      const {loss,grad} = computeLossAndGrad(X,y,weights);
      // record current
      history.push({iter:it,loss,weights:weights.slice()});
      // update weights: w = w - lr * grad
      for(let j=0;j<D;j++) weights[j] = weights[j] - opts.learningRate * grad[j];
    }
    // final record
    const {loss:finalLoss} = computeLossAndGrad(X,y,weights);
    history.push({iter:opts.iterations,loss:finalLoss,weights:weights.slice()});
    return history;
  }

  function historyToCSV(history){
    if(history.length===0) return '';
    const D = history[0].weights.length;
    const header = ['iter','loss'];
    for(let j=0;j<D;j++) header.push('w'+j);
    const lines = [header.join(',')];
    for(const h of history){
      const vals = [h.iter, h.loss.toFixed(10), ...h.weights.map(w=>w.toFixed(10))];
      lines.push(vals.join(','));
    }
    return lines.join('\n');
  }

  function log(msg){
    logEl.textContent += msg + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  trainBtn.addEventListener('click', ()=>{
    outCsv.value = '';
    logEl.textContent = '';
    downloadBtn.disabled = true;
    try{
      const parsed = parseCSV(csvInput.value);
      if(parsed.X.length===0) { log('No valid rows found in CSV'); return; }
      const prepared = prepareData(parsed);
      const lr = Number(lrInput.value);
      const iters = parseInt(iterInput.value,10)||100;
      const init = initSelect.value;
      log(`Training on ${prepared.X.length} samples, ${prepared.X[0].length} features (including bias)`);
      const history = train(prepared.X, prepared.y, {learningRate:lr,iterations:iters,init:init});
      const csv = historyToCSV(history);
      outCsv.value = csv;
      log('Training complete. You can download the CSV of weight updates.');
      downloadBtn.disabled = false;
      // attach data to button
      downloadBtn._csv = csv;
    }catch(err){
      log('Error: '+err.message);
    }
  });

  downloadBtn.addEventListener('click', ()=>{
    const csv = downloadBtn._csv || outCsv.value;
    if(!csv) return;
    const blob = new Blob([csv], {type: 'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'weight_history.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

});
