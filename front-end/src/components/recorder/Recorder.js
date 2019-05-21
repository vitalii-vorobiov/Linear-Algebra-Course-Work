import React, {Component} from "react";
import Webcam from "react-webcam";
import "./Recorder.css"

class Recorder extends Component {
    constructor(props) {
        super(props);
        this.state = {
            name: "Recognize..."
        }
    }

    setRef = webcam => {
        this.webcam = webcam;
    };

    capture = async () => {
        const imageSrc = this.webcam.getScreenshot();

        const settings = {
            method: 'POST',
            body: JSON.stringify(imageSrc),
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            }
        };

        const data = await fetch(`http://127.0.0.1:5000/recognize`, settings)
            .then(response => response.json())
            .then(json => {
                return json;
            })
            .catch(e => {
                return e
            });

        this.setState({name: data});

    //     window.fetch("http://127.0.0.1:5000/recognize", {
    //         method: 'POST',
    //         body: JSON.stringify(imageSrc),
    //         headers: {
    //             'Content-Type': 'application/json'
    //         }
    // }).then(res => console.log(res.json()))
    // .then(res => console.log('Success:', JSON.stringify(res)))
    // .catch(error => console.error('Error:', error));
    };

    render() {
        return (
            <div className="Recorder">
                <Webcam
                    className="Recorder__video"
                    audio={false}
                    ref={this.setRef}
                    screenshotFormat="image/jpeg"
                />
                <button className="Recorder__button" onClick={this.capture}>Capture photo</button>
                <p className="Recorder__person-name">{this.state.name}</p>
            </div>
        );
    }
}

export default Recorder;