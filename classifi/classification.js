document.addEventListener('DOMContentLoaded', ()=>{
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const rect = () => canvas.getBoundingClientRect();

  const sigmoidCanvas = document.getElementById('sigmoid-canvas');
  const sigmoidCtx = sigmoidCanvas.getContext('2d');

  const btnRandom = document.getElementById('btn-random');
  const btnClear = document.getElementById('btn-clear');
  const btnTrain = document.getElementById('btn-train');
  const btnStop = document.getElementById('btn-stop');
  const epochsInput = document.getElementById('epochs');
  // fixed base learning rate (auto-decay always enabled)
  const baseLr = 0.05;
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

  function drawSigmoidViz(){
    sigmoidCtx.clearRect(0, 0, sigmoidCanvas.width, sigmoidCanvas.height);
    const padding = 40;
    const w = sigmoidCanvas.width - 2*padding;
    const topY = padding + 12;
    const bottomY = sigmoidCanvas.height - padding - 12;

    // vertical decision line where z = 0 => nx = -b/wx
    const b = weights[0], wx = weights[1];
    let nx0 = 1e9;
    if(Math.abs(wx) > 1e-8) nx0 = -b / wx;
    const px0 = padding + (Math.max(-1, Math.min(1, nx0)) + 1)/2 * w;

    // shade left and right of decision line based on predicted class
    sigmoidCtx.fillStyle = 'rgba(239,68,68,0.08)';
    sigmoidCtx.fillRect(padding, 0, px0 - padding, sigmoidCanvas.height);
    sigmoidCtx.fillStyle = 'rgba(22,163,74,0.08)';
    sigmoidCtx.fillRect(px0, 0, sigmoidCanvas.width - padding - px0, sigmoidCanvas.height);

    // background regions: top = positive (green), bottom = negative (red)
    sigmoidCtx.fillStyle = 'rgba(22,163,74,0.06)';
    sigmoidCtx.fillRect(padding, 0, w, topY + 8);
    sigmoidCtx.fillStyle = 'rgba(239,68,68,0.06)';
    sigmoidCtx.fillRect(padding, bottomY - 8, w, sigmoidCanvas.height - (bottomY - 8));

    // draw sigmoid curve as a function of normalized x (nx) using b + w_x * nx
    sigmoidCtx.strokeStyle = 'rgba(255,255,255,0.95)';
    sigmoidCtx.lineWidth = 2;
    const samples = 200;
    sigmoidCtx.beginPath();
    for(let i=0;i<=samples;i++){
      const nx = -1 + (2*i)/samples; // domain of normalized x
      const z = b + wx * nx; // ignore wy*ny here so curve shows how x-axis aligns
      const prob = sigmoid(z);
      const px = padding + (nx + 1)/2 * w;
      const py = topY + (1 - prob) * (bottomY - topY);
      if(i===0) sigmoidCtx.moveTo(px, py); else sigmoidCtx.lineTo(px, py);
    }
    sigmoidCtx.stroke();

    // vertical decision line where z = 0
    sigmoidCtx.strokeStyle = 'rgba(255,200,100,0.95)';
    sigmoidCtx.lineWidth = 2;
    sigmoidCtx.beginPath();
    sigmoidCtx.moveTo(px0, topY - 6);
    sigmoidCtx.lineTo(px0, bottomY + 6);
    sigmoidCtx.stroke();
    sigmoidCtx.lineWidth = 1;

    // draw points at fixed horizontal positions based on their normalized x (nx)
    for(const p of points){
      const nx = normalizeX(p.x);
      const px = padding + (nx + 1)/2 * w;
      const py = p.label > 0 ? topY : bottomY;
      sigmoidCtx.beginPath();
      sigmoidCtx.fillStyle = p.label > 0 ? 'rgba(22,163,74,0.95)' : 'rgba(239,68,68,0.95)';
      sigmoidCtx.arc(px, py, 5, 0, Math.PI*2);
      sigmoidCtx.fill();
      sigmoidCtx.strokeStyle = 'rgba(0,0,0,0.25)';
      sigmoidCtx.stroke();
    }

    // labels
    sigmoidCtx.fillStyle = 'rgba(255,255,255,0.8)';
    sigmoidCtx.font = '12px sans-serif';
    sigmoidCtx.textAlign = 'center';
    sigmoidCtx.fillText('Logit (z = b + w₁x + w₂y)', sigmoidCanvas.width/2, sigmoidCanvas.height - 10);
    sigmoidCtx.textAlign = 'left';
    sigmoidCtx.fillText('Positive (y=1)', padding, topY - 18);
    sigmoidCtx.fillText('Negative (y=0)', padding, bottomY + 22);
  }

  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // grid
    ctx.strokeStyle='rgba(255,255,255,0.03)';
    for(let x=0;x<canvas.width;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
    for(let y=0;y<canvas.height;y+=50){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}

    // probability-based shading (draw before points)
    {
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
    if(points.length === 0){ accuracyEl.textContent = 'Accuracy: —'; epochEl.textContent = 'Epoch: 0'; currentLrEl.textContent = `LR: ${baseLr.toFixed(4)}`; return; }
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
    epochsInput.disabled = disabled;
  }

  function trainLogisticAnimated(initialLr=0.05, epochs=20, delay=200){
    if(points.length === 0) return;
    if(trainingInterval) return;
    let currentEpoch = 0;
    disableUI(true);
    btnStop.disabled = false;

    trainingInterval = setInterval(()=>{
      // current learning rate schedule (auto-decay always enabled)
      const lr = initialLr / (1 + currentEpoch * 0.05);
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
      drawSigmoidViz();
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
    drawSigmoidViz();
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
    drawSigmoidViz();
  });

  btnClear.addEventListener('click', ()=>{
    points = [];
    weights = [Math.random()*0.2-0.1, Math.random()*0.2-0.1, Math.random()*0.2-0.1];
    updateAccuracy();
    draw();
    drawSigmoidViz();
  });

  btnTrain.addEventListener('click', ()=>{
    const epochs = parseInt(epochsInput.value) || 20;
    trainLogisticAnimated(baseLr, epochs, 160);
  });

  // initial draw
  updateAccuracy();
  draw();
  drawSigmoidViz();
});
