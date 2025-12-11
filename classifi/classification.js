document.addEventListener('DOMContentLoaded', ()=>{
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const rect = () => canvas.getBoundingClientRect();

  const btnRandom = document.getElementById('btn-random');
  const btnClear = document.getElementById('btn-clear');
  const btnTrain = document.getElementById('btn-train');
  const btnStop = document.getElementById('btn-stop');
  const lrInput = document.getElementById('lr');
  const epochsInput = document.getElementById('epochs');
  const autoLrInput = document.getElementById('auto-lr');
  const accuracyEl = document.getElementById('accuracy');
  const epochEl = document.getElementById('epoch');
  const currentLrEl = document.getElementById('current-lr');
  const showBoundary = document.getElementById('show-boundary');

  let points = [];
  // logistic weights: [bias, w_x, w_y]
  let weights = [Math.random()*0.2-0.1, Math.random()*0.2-0.1, Math.random()*0.2-0.1];
  let trainingInterval = null;

  function normalizeX(x){ return (x - 0) / canvas.width * 2 - 1; }
  function normalizeY(y){ return (y - 0) / canvas.height * -2 + 1; }
  function denormX(nx){ return (nx + 1) / 2 * canvas.width; }
  function denormY(ny){ return (1 - ny) / 2 * canvas.height; }

  function sigmoid(z){
    const v = Math.max(-30, Math.min(30, z));
    return 1 / (1 + Math.exp(-v));
  }

  function predictProb(nx, ny){
    const sum = weights[0] + weights[1]*nx + weights[2]*ny;
    return sigmoid(sum);
  }

  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // grid
    ctx.strokeStyle='rgba(255,255,255,0.03)';
    for(let x=0;x<canvas.width;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
    for(let y=0;y<canvas.height;y+=50){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}

    // probability-based shading (draw before points)
    if(showBoundary.checked){
      const step = 8;
      for(let sx = 0; sx < canvas.width; sx += step){
        for(let sy = 0; sy < canvas.height; sy += step){
          const nx = normalizeX(sx + step/2);
          const ny = normalizeY(sy + step/2);
          const p = predictProb(nx, ny);
          if(p >= 0.5){
            const a = (p - 0.5)/0.5 * 0.20; // stronger for confident
            ctx.fillStyle = `rgba(22,163,74,${a})`;
          } else {
            const a = (0.5 - p)/0.5 * 0.12;
            ctx.fillStyle = `rgba(239,68,68,${a})`;
          }
          ctx.fillRect(sx, sy, step, step);
        }
      }
      // draw decision contour (approximate) by sampling border between cells
      ctx.strokeStyle='rgba(255,255,255,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      // draw a simple approximate line by sampling x positions and computing y where p=0.5
      const samples = 120;
      let started = false;
      for(let i=0;i<=samples;i++){
        const nx = -1 + (2*i)/samples;
        // solve for y such that sigmoid(b + wx*nx + wy*ny)=0.5 => b + wx*nx + wy*ny = 0 => ny = -(b + wx*nx)/wy
        const b = weights[0], wx = weights[1], wy = weights[2];
        let ny;
        if(Math.abs(wy) < 1e-8){ ny = 0; }
        else ny = -(b + wx*nx)/wy;
        const px = denormX(nx); const py = denormY(ny);
        if(!started){ ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
      }
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    // draw points
    for(const p of points){
      ctx.beginPath();
      ctx.fillStyle = p.label > 0 ? '#16a34a' : '#ef4444';
      ctx.arc(p.x, p.y, 6, 0, Math.PI*2);
      ctx.fill();
      ctx.strokeStyle='rgba(0,0,0,0.25)';
      ctx.stroke();
    }
  }

  function updateAccuracy(){
    if(points.length === 0){ accuracyEl.textContent = 'Accuracy: —'; epochEl.textContent = 'Epoch: 0'; currentLrEl.textContent = `LR: ${parseFloat(lrInput.value).toFixed(4)}`; return; }
    let ok=0;
    for(const p of points){
      const nx = normalizeX(p.x), ny = normalizeY(p.y);
      const prob = predictProb(nx, ny);
      const pred = prob >= 0.5 ? 1 : -1;
      if(pred === p.label) ok++;
    }
    const acc = Math.round((ok / points.length) * 100);
    accuracyEl.textContent = `Accuracy: ${acc}%`;
  }

  function disableUI(disabled){
    btnTrain.disabled = disabled;
    btnRandom.disabled = disabled;
    btnClear.disabled = disabled;
    lrInput.disabled = disabled;
    epochsInput.disabled = disabled;
    autoLrInput.disabled = disabled;
  }

  function trainLogisticAnimated(initialLr=0.05, epochs=20, delay=200){
    if(points.length === 0) return;
    if(trainingInterval) return;
    let currentEpoch = 0;
    disableUI(true);
    btnStop.disabled = false;

    trainingInterval = setInterval(()=>{
      // current learning rate schedule
      const lr = autoLrInput.checked ? initialLr / (1 + currentEpoch * 0.05) : initialLr;
      currentLrEl.textContent = `LR: ${lr.toFixed(4)}`;

      // one epoch of SGD (sequential)
      for(const p of points){
        const nx = normalizeX(p.x), ny = normalizeY(p.y);
        const y = p.label > 0 ? 1 : 0; // logistic uses 0/1 labels
        const pred = predictProb(nx, ny);
        const grad = (y - pred); // gradient of log-likelihood
        weights[0] += lr * grad * 1; // bias
        weights[1] += lr * grad * nx;
        weights[2] += lr * grad * ny;
      }

      currentEpoch++;
      updateAccuracy();
      epochEl.textContent = `Epoch: ${currentEpoch}/${epochs}`;
      draw();
      if(currentEpoch >= epochs){
        clearInterval(trainingInterval);
        trainingInterval = null;
        disableUI(false);
        btnStop.disabled = true;
      }
    }, delay);
  }

  btnStop.addEventListener('click', ()=>{
    if(trainingInterval){
      clearInterval(trainingInterval);
      trainingInterval = null;
      disableUI(false);
      btnStop.disabled = true;
      epochEl.textContent += ' (stopped)';
    }
  });

  // gaussian random (Box-Muller)
  function randnBM(){
    let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
    return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v);
  }

  canvas.addEventListener('click', (ev)=>{
    const r = rect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    const label = ev.shiftKey ? -1 : 1;
    points.push({x,y,label});
    updateAccuracy();
    draw();
  });

  btnRandom.addEventListener('click', ()=>{
    // produce overlapping clusters with moderate noise (a bit easier to classify)
    const centers = [
      {x:150, y:140, label:1},
      {x:350, y:240, label:-1}
    ];
    const n = 70;
    for(let i=0;i<n;i++){
      const c = centers[Math.random() < 0.5 ? 0 : 1];
      const x = c.x + randnBM()*48; // slightly smaller spread
      const y = c.y + randnBM()*48;
      // label noise reduced
      let label = c.label;
      if(Math.random() < 0.10) label = -label;
      // keep inside canvas
      const px = Math.max(6, Math.min(canvas.width-6, x));
      const py = Math.max(6, Math.min(canvas.height-6, y));
      points.push({x:px, y:py, label});
    }
    updateAccuracy();
    draw();
  });

  btnClear.addEventListener('click', ()=>{
    points = [];
    weights = [Math.random()*0.2-0.1, Math.random()*0.2-0.1, Math.random()*0.2-0.1];
    updateAccuracy();
    draw();
  });

  btnTrain.addEventListener('click', ()=>{
    const lr = parseFloat(lrInput.value) || 0.05;
    const epochs = parseInt(epochsInput.value) || 20;
    trainLogisticAnimated(lr, epochs, 160);
  });

  // initial draw
  updateAccuracy();
  draw();
});
