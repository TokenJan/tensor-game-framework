import React from 'react';
import './Panel.css';
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

export default class Panel extends React.Component {

    webcam : any;
    model : tf.LayersModel;
  
    state = {
      isPredicting : false,
    }
  
    componentDidMount() {
      this.init();
    }
  
    async init() {
      try {
        this.webcam = await tfd.webcam(document.getElementById("webcam") as HTMLVideoElement);
      } catch (e) {
        console.log(e);
      }
      this.model = await this.loadModel();
      const screetshot = await this.webcam.capture();
      this.model.predict(screetshot.expandDims(0));
      screetshot.dispose();
    }
  
    async loadModel() {
      console.log('Loading model......');
      try {
        const model = await tf.loadLayersModel("./tfjs_model/model.json", {strict: false});
        document.getElementById("status").style.display = "none";
        document.getElementById("main").style.display = "flex";
        console.log('Done loading pretrained model.');
        return model;
      } catch (err) {
        console.log(err.message);
      }
    }
  
    async getImage() {
      const img = await this.webcam.capture();
      const processedImg =
          tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
      img.dispose();
      return processedImg;    
    }
  
    async predict() {
      // Capture the frame from the webcam.
      const img = await this.getImage();
      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model, i.e., "embeddings" of the input images.
      const predictions = this.model.predict(img) as tf.Tensor;
  
      console.log(predictions.as1D().dataSync());
      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      const predictedClass = predictions.as1D().argMax();
      const classId = (await predictedClass.data())[0];
      this.handleButton(classId);
      img.dispose();
      await tf.nextFrame();
    }
  
    handleButton(label: number) {
      const CONTROLS = ['up-btn', 'down-btn', 'left-btn', 'right-btn'];
      const button = document.getElementById(CONTROLS[label]);
      button.classList.add("is-primary");
    }
  
    play = () => {
      this.setState({
        isPredicting : true,
      });
  
      setInterval(() => {this.predict()}, 1000)
    }

    render() {
        return(
            <div id="game-panel">
                <div className="nes-field">
                    <label htmlFor="name_field" id="name">Your name</label>
                    <input type="text" id="name_field" className="nes-input"/>
                </div>
                <div id="webcam-controller">
                    <video autoPlay playsInline muted id="webcam" width="224" height="224"></video>
                <div id="direction-panel">
                    <div id="top-panel">
                        <span className="fill"></span>
                        <button className="nes-btn scroll-btn top" id="top-btn"><span>&lt;</span></button>
                    </div>
                    <div id="middle-panel">
                        <button className="nes-btn left" id="left-btn"><span>&lt;</span></button>
                        <span className="fill"></span>
                        <button className="nes-btn" id="right-btn"><span>&gt;</span></button>
                    </div>
                    <div id="bottom-panel">
                        <span className="fill"></span>
                        <button className="nes-btn scroll-btn" id="bottom-btn"><span>&gt;</span></button>
                    </div>
                </div>
                </div>
                <button className="nes-btn" onClick={this.play} id="play">Play</button>
            </div>

        )
    }
}