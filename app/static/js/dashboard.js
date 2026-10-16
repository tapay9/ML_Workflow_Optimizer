// Model Performance Chart

const performanceContainer = d3.select("#performance-chart");

const performanceMargin = {
    top: 30,
    right: 30,
    bottom: 70,
    left: 60
};

const performanceWidth = 700;
const performanceHeight = 300;

const performanceSvg = performanceContainer
    .append("svg")
    .attr("viewBox", `0 0 ${performanceWidth} ${performanceHeight}`)
    .attr("width", "100%");


const metrics = [
    {
        name: "Accuracy",
        key: "test_acc"
    },
    {
        name: "Macro-F1",
        key: "macro_f1"
    },
    {
        name: "ROC-AUC",
        key: "roc_auc"
    }
];


const x0 = d3.scaleBand()
    .domain(modelData.map(d => d.model))
    .range([
        performanceMargin.left,
        performanceWidth - performanceMargin.right
    ])
    .padding(0.25);


const x1 = d3.scaleBand()
    .domain(metrics.map(d => d.name))
    .range([0, x0.bandwidth()])
    .padding(0.08);


const y = d3.scaleLinear()
    .domain([0, 1])
    .range([
        performanceHeight - performanceMargin.bottom,
        performanceMargin.top
    ]);


const color = d3.scaleOrdinal()
    .domain(metrics.map(d => d.name))
    .range([
        "#7867ff",
        "#55dca5",
        "#5da9ff"
    ]);


performanceSvg
    .append("g")
    .attr(
        "transform",
        `translate(0, ${performanceHeight - performanceMargin.bottom})`
    )
    .call(d3.axisBottom(x0))
    .selectAll("text")
    .attr("fill", "#9aa5b5")
    .attr("font-size", "11px");


performanceSvg
    .append("g")
    .attr(
        "transform",
        `translate(${performanceMargin.left}, 0)`
    )
    .call(
        d3.axisLeft(y)
            .ticks(5)
            .tickFormat(d3.format(".0%"))
    )
    .selectAll("text")
    .attr("fill", "#9aa5b5");


performanceSvg
    .selectAll(".domain")
    .attr("stroke", "#273244");

performanceSvg
    .selectAll(".tick line")
    .attr("stroke", "#273244");


const modelGroups = performanceSvg
    .selectAll(".model-group")
    .data(modelData)
    .enter()
    .append("g")
    .attr("class", "model-group")
    .attr(
        "transform",
        d => `translate(${x0(d.model)}, 0)`
    );


modelGroups
    .selectAll("rect")
    .data(d =>
        metrics.map(metric => ({
            metric: metric.name,
            value: Number(d[metric.key])
        }))
    )
    .enter()
    .append("rect")
    .attr("x", d => x1(d.metric))
    .attr("y", d => y(d.value))
    .attr("width", x1.bandwidth())
    .attr(
        "height",
        d => y(0) - y(d.value)
    )
    .attr("rx", 3)
    .attr("fill", d => color(d.metric));


const legend = performanceSvg
    .append("g")
    .attr("transform", "translate(60, 275)");


metrics.forEach((metric, index) => {

    const item = legend
        .append("g")
        .attr(
            "transform",
            `translate(${index * 120}, 0)`
        );

    item
        .append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("rx", 2)
        .attr("fill", color(metric.name));

    item
        .append("text")
        .attr("x", 16)
        .attr("y", 9)
        .attr("fill", "#9aa5b5")
        .attr("font-size", "11px")
        .text(metric.name);
});


// Training vs Test Accuracy Chart

const overfitContainer = d3.select("#overfitting-chart");

const overfitMargin = {
    top: 30,
    right: 30,
    bottom: 70,
    left: 60
};

const overfitWidth = 700;
const overfitHeight = 300;


const overfitSvg = overfitContainer
    .append("svg")
    .attr(
        "viewBox",
        `0 0 ${overfitWidth} ${overfitHeight}`
    )
    .attr("width", "100%");


const accuracyTypes = [
    {
        name: "Train Accuracy",
        key: "train_acc"
    },
    {
        name: "Test Accuracy",
        key: "test_acc"
    }
];


const overfitX0 = d3.scaleBand()
    .domain(modelData.map(d => d.model))
    .range([
        overfitMargin.left,
        overfitWidth - overfitMargin.right
    ])
    .padding(0.25);


const overfitX1 = d3.scaleBand()
    .domain(accuracyTypes.map(d => d.name))
    .range([0, overfitX0.bandwidth()])
    .padding(0.08);


const overfitY = d3.scaleLinear()
    .domain([0, 1])
    .range([
        overfitHeight - overfitMargin.bottom,
        overfitMargin.top
    ]);


const overfitColor = d3.scaleOrdinal()
    .domain(accuracyTypes.map(d => d.name))
    .range([
        "#7867ff",
        "#55dca5"
    ]);


overfitSvg
    .append("g")
    .attr(
        "transform",
        `translate(0, ${overfitHeight - overfitMargin.bottom})`
    )
    .call(d3.axisBottom(overfitX0))
    .selectAll("text")
    .attr("fill", "#9aa5b5")
    .attr("font-size", "11px");


overfitSvg
    .append("g")
    .attr(
        "transform",
        `translate(${overfitMargin.left}, 0)`
    )
    .call(
        d3.axisLeft(overfitY)
            .ticks(5)
            .tickFormat(d3.format(".0%"))
    )
    .selectAll("text")
    .attr("fill", "#9aa5b5");


overfitSvg
    .selectAll(".domain")
    .attr("stroke", "#273244");

overfitSvg
    .selectAll(".tick line")
    .attr("stroke", "#273244");


const overfitGroups = overfitSvg
    .selectAll(".overfit-group")
    .data(modelData)
    .enter()
    .append("g")
    .attr("class", "overfit-group")
    .attr(
        "transform",
        d => `translate(${overfitX0(d.model)}, 0)`
    );


overfitGroups
    .selectAll("rect")
    .data(d =>
        accuracyTypes.map(type => ({
            type: type.name,
            value: Number(d[type.key])
        }))
    )
    .enter()
    .append("rect")
    .attr("x", d => overfitX1(d.type))
    .attr("y", d => overfitY(d.value))
    .attr("width", overfitX1.bandwidth())
    .attr(
        "height",
        d => overfitY(0) - overfitY(d.value)
    )
    .attr("rx", 3)
    .attr("fill", d => overfitColor(d.type));


const overfitLegend = overfitSvg
    .append("g")
    .attr("transform", "translate(60, 275)");


accuracyTypes.forEach((type, index) => {

    const item = overfitLegend
        .append("g")
        .attr(
            "transform",
            `translate(${index * 140}, 0)`
        );

    item
        .append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("rx", 2)
        .attr("fill", overfitColor(type.name));

    item
        .append("text")
        .attr("x", 16)
        .attr("y", 9)
        .attr("fill", "#9aa5b5")
        .attr("font-size", "11px")
        .text(type.name);
});