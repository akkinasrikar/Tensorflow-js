let x_values=[]
let y_values=[]
let m,b;

const learningRate=0.1;
const optimizer = tf.train.sgd(learningRate)

function setup(){
	createCanvas(1080,800);
	background(0);
	m=tf.variable(tf.scalar(random(1)));
	b=tf.variable(tf.scalar(random(1)));
}

function loss(pred,labels){
	//(pred,labels)==>pred.sub(label).square().mean();
    return pred.sub(labels).square().mean();
}


function predict(x){
	const xs=tf.tensor1d(x);
	const ys=xs.mul(m).add(b);

	return ys;
}

function mousePressed() {
	
	let x=map(mouseX, 0, width, 0, 1);
	let y=map(mouseY, 0, height, 1, 0);
	x_values.push(x);
	y_values.push(y);
}

function draw(){
    
    if(x_values.length>0){
    const ys=tf.tensor1d(y_values);
	optimizer.minimize(() => loss(predict(x_values),ys));
	}

	background(0);
	stroke(0,160,255);
	strokeWeight(13);
	for(let i=0; i<x_values.length; i++){
		let px=map(x_values[i], 0,1,0, width);
		let py=map(y_values[i], 0,1,height,0);
		point(px, py);
	}

  tf.tidy(()=>{
  const xs = [0, 1];
  const ys=predict(xs);

  let x1 = map(xs[0], 0, 1, 0, width);
  let x2 = map(xs[1], 0, 1, 0, width);

  let liney = ys.dataSync();
  ys.dispose();

  let y1 = map(liney[0], 0, 1, height, 0);
  let y2 = map(liney[1], 0, 1, height, 0);
  
  stroke(255);
  strokeWeight(8);
  line(x1, y1, x2, y2);
});

  console.log(tf.memory().numTensors);
  //noLoop();
}


