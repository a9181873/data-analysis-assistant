import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { Streamlit } from "streamlit-component-lib";
import MLMetricsCard from "./MLMetricsCard";
import "./index.css";

const StreamlitWrapper = () => {
  const [args, setArgs] = useState(null);

  useEffect(() => {
    const onRender = (event) => {
      setArgs(event.detail.args);
      Streamlit.setFrameHeight();
    };

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight();

    return () => {
      Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
    };
  }, []);

  if (!args) {
    return <div className="text-white">Loading...</div>;
  }

  return (
    <MLMetricsCard 
      accuracy={args.accuracy} 
      metrics={args.metrics} 
    />
  );
};

// Start rendering
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <StreamlitWrapper />
  </React.StrictMode>
);
