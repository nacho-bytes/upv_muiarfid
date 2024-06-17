import * as THREE from "../lib/three.module.js";
import { OrbitControls } from "../lib/OrbitControls.module.js";
import {GUI} from "../lib/lil-gui.module.min.js";

let canvas, scene, camera, renderer, cube;

const RED = 0xff0000;
const GREEN = 0x00ff00;
const BLUE = 0x0000ff;
const LIMIT = 100;

const speed = new THREE.Vector3();

function initCamera() {
  camera = new THREE.PerspectiveCamera(70, 1, 1, 10000);
  scene.add(camera);
  camera.position.set(90, 80, 70);
}

function initRenderer() {
   // Instanciar el motor de render
   renderer = new THREE.WebGLRenderer();
   renderer.setSize(window.innerWidth,window.innerHeight);
   document.getElementById('container').appendChild( renderer.domElement );
}

function render() {
  cube.position.add(speed);
  //flip direction of component if >LIMIT
  for(let i=0; i<3; i++){
    if (((speed.getComponent(i)>0) && (cube.position.getComponent(i) > LIMIT)) || ((speed.getComponent(i) < 0) && (cube.position.getComponent(i) < -LIMIT))) {
      speed.setComponent(i, -speed.getComponent(i)); 
    } 
  }
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

function initAxes() {
  function makeAxis(start, finish, material, name) {
    const points = [start, finish];
    const axisGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const newAxis = new THREE.Line(axisGeometry, material);
    newAxis.name = name;
    scene.add(newAxis);
  }
  const xAxisMaterial = new THREE.LineBasicMaterial({
    color: RED
  });
  const yAxisMaterial = new THREE.LineBasicMaterial({
    color: GREEN
  });
  const zAxisMaterial = new THREE.LineBasicMaterial({
    color: BLUE
  });
  makeAxis(new THREE.Vector3(-LIMIT, 0, 0), new THREE.Vector3(LIMIT, 0, 0), xAxisMaterial, 'xAxis');
  makeAxis(new THREE.Vector3(0, -LIMIT, 0), new THREE.Vector3(0, LIMIT, 0), yAxisMaterial, 'yAxis');
  makeAxis(new THREE.Vector3(0, 0, -LIMIT), new THREE.Vector3(0, 0, LIMIT), zAxisMaterial, 'zAxis');
}

function initCube() {
  const geometry = new THREE.BoxGeometry(10, 10, 10);
  const material = new THREE.MeshPhongMaterial({
    color: 0xff0000,
  });
  cube = new THREE.Mesh(geometry, material);
  scene.add(cube);
}
function initGui(){
  const gui = new GUI();
  const cubePositionFolder  = gui.addFolder('cube')
  cubePositionFolder.add(cube.position, 'x', -LIMIT, LIMIT).listen();
  cubePositionFolder.add(cube.position, 'y', -LIMIT, LIMIT).listen();
  cubePositionFolder.add(cube.position, 'z', -LIMIT, LIMIT).listen();
  const cubeSpeedFolder  = gui.addFolder('speed')
  cubeSpeedFolder.add({vx:0}, 'vx', [0,1,2]).name('xSpeed').onChange(function(value){speed.x= Number(value);});
  cubeSpeedFolder.add({vy:0}, 'vy', [0,1,2]).name('ySpeed').onChange(function(value){speed.y= Number(value);});
  cubeSpeedFolder.add({vz:0}, 'vz', [0,1,2]).name('zSpeed').onChange(function(value){speed.z= Number(value);});
}
window.onload = function() {
  canvas = document.getElementById('canvas');
  scene = new THREE.Scene();
  initCamera();
  initRenderer();
  initAxes();
  // Control de camara
  const controls = new OrbitControls(camera, renderer.domElement);  
  initCube();
  scene.add(new THREE.AmbientLight(0xffffff));
  initGui();
  render();
}
