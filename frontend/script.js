const form = document.getElementById("predictForm");
const jsonPreview = document.getElementById("jsonPreview");
const updateJsonButton = document.getElementById("updateJsonButton");
const copyJsonButton = document.getElementById("copyJsonButton");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

const numericFields = [
  "Age",
  "Daily_Social_Media_Hours",
  "Screen_Time_Hours",
  "Night_Scrolling_Frequency",
  "Online_Gaming_Hours",
  "Exercise_Frequency_per_Week",
  "Daily_Sleep_Hours",
  "Caffeine_Intake_Cups",
  "Study_Work_Hours_per_Day",
  "Overthinking_Score",
  "Anxiety_Score",
  "Mood_Stability_Score",
  "Social_Comparison_Index",
  "Sleep_Quality_Score",
  "Motivation_Level",
  "Emotional_Fatigue_Score",
  "Wellbeing_Index",
];

function getInputValueById(id) {
  const element = document.getElementById(id);

  if (!element) {
    console.error(`Campo não encontrado no HTML: ${id}`);
    return 0;
  }

  return Number(element.value);
}

function getGenderDummies(selectedGender) {
  return {
    "Gender_Female": selectedGender === "Female",
    "Gender_Male": selectedGender === "Male",
    "Gender_Non-binary": selectedGender === "Non-binary",
  };
}

function getStatusDummies(selectedStatus) {
  return {
    "Student_Working_Status_Both": selectedStatus === "Both",
    "Student_Working_Status_Student": selectedStatus === "Student",
    "Student_Working_Status_Working": selectedStatus === "Working",
  };
}

function getContentPreferenceDummies(selectedContent) {
  return {
    "Content_Type_Preference_Educational": selectedContent === "Educational",
    "Content_Type_Preference_Entertainment": selectedContent === "Entertainment",
    "Content_Type_Preference_Gaming": selectedContent === "Gaming",
    "Content_Type_Preference_Lifestyle": selectedContent === "Lifestyle",
    "Content_Type_Preference_News": selectedContent === "News",
  };
}

function getNumericFeatures() {
  const features = {};

  numericFields.forEach((fieldName) => {
    features[fieldName] = getInputValueById(fieldName);
  });

  return features;
}

function getSelectedValue(id) {
  const element = document.getElementById(id);

  if (!element) {
    console.error(`Dropdown não encontrado no HTML: ${id}`);
    return "";
  }

  return element.value;
}

function getFormFeatures() {
  const selectedGender = getSelectedValue("Gender");
  const selectedStatus = getSelectedValue("Student_Working_Status");
  const selectedContent = getSelectedValue("Content_Type_Preference");

  const features = {
    ...getNumericFeatures(),
    ...getGenderDummies(selectedGender),
    ...getStatusDummies(selectedStatus),
    ...getContentPreferenceDummies(selectedContent),
  };

  return {
    features,
  };
}

function updateJsonPreview() {
  if (!jsonPreview) {
    console.error("Textarea jsonPreview não encontrada no HTML.");
    return;
  }

  const payload = getFormFeatures();
  jsonPreview.value = JSON.stringify(payload, null, 2);
}

function setStatus(message, type = "") {
  statusBox.textContent = message;
  statusBox.className = "status";

  if (type) {
    statusBox.classList.add(type);
  }
}

async function copyJson() {
  await navigator.clipboard.writeText(jsonPreview.value);
  setStatus("JSON copiado para a área de transferência.", "success");
}

async function sendPrediction(event) {
  event.preventDefault();

  updateJsonPreview();

  let payload;

  try {
    payload = JSON.parse(jsonPreview.value);
  } catch (error) {
    setStatus("JSON inválido. Atualize ou corrija o JSON antes de enviar.", "error");
    resultBox.textContent = String(error);

    document.getElementById("predictionResult").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });

    return;
  }

  setStatus("Enviando dados para /predict...");
  resultBox.textContent = "Aguardando resposta da API...";

  document.getElementById("predictionResult").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      setStatus("Erro na predição.", "error");
      resultBox.textContent = JSON.stringify(data, null, 2);
      return;
    }

    setStatus("Predição realizada com sucesso.", "success");
    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    setStatus("Erro ao conectar com a API.", "error");
    resultBox.textContent = String(error);
  }
}

updateJsonButton.addEventListener("click", updateJsonPreview);
copyJsonButton.addEventListener("click", copyJson);
form.addEventListener("input", updateJsonPreview);
form.addEventListener("change", updateJsonPreview);
form.addEventListener("submit", sendPrediction);

document.addEventListener("DOMContentLoaded", updateJsonPreview);
updateJsonPreview();