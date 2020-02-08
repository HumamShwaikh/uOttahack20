import React, { Component } from "react";
import { Button } from "semantic-ui-react";
import Home from "./Home";
import "./LandingPage.css";
import logo from "./styles/snails_PNG13216.png";

class App extends Component {
  render() {
    return (
      <React.Fragment>
        <div>
          {/* <img src={logo}></img> */}
          <Button basic color="pink">
            Fear Death
          </Button>
        </div>
      </React.Fragment>
    );
  }
}

const center = {
  display: "block",
  marginLeft: "auto",
  marginRight: "auto",
  width: "0",
  height: "auto"
};

export default App;
