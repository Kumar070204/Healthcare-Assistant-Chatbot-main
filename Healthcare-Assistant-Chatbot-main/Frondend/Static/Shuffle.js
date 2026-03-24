var Shuffle = (function () {
  var gsap = window.gsap;
  var ScrollTrigger = window.ScrollTrigger;
  var GSAPSplitText = window.SplitText;

  gsap.registerPlugin(ScrollTrigger, GSAPSplitText);

  return function Shuffle({
    text,
    className = '',
    style = {},
    shuffleDirection = 'right',
    duration = 0.35,
    maxDelay = 0,
    ease = 'power3.out',
    threshold = 0.1,
    rootMargin = '-100px',
    tag = 'p',
    textAlign = 'center',
    onShuffleComplete,
    shuffleTimes = 1,
    animationMode = 'evenodd',
    loop = false,
    loopDelay = 0,
    stagger = 0.03,
    scrambleCharset = '',
    colorFrom,
    colorTo,
    triggerOnce = true,
    respectReducedMotion = true,
    triggerOnHover = true
  }) {
    var ref = { current: null };
    var fontsLoaded = false;
    var ready = false;
    var splitRef = { current: null };
    var wrappersRef = { current: [] };
    var tlRef = { current: null };
    var playingRef = { current: false };
    var hoverHandlerRef = { current: null };

    if ('fonts' in document) {
      if (document.fonts.status === 'loaded') fontsLoaded = true;
      else document.fonts.ready.then(function () { fontsLoaded = true; });
    } else fontsLoaded = true;

    function runAnimation() {
      if (!ref.current || !text || !fontsLoaded) return;
      if (respectReducedMotion && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        onShuffleComplete?.();
        return;
      }
      var el = ref.current;
      var startPct = (1 - threshold) * 100;
      var mm = /^(-?\d+(?:\.\d+)?)(px|em|rem|%)?$/.exec(rootMargin || '');
      var mv = mm ? parseFloat(mm[1]) : 0;
      var mu = mm ? mm[2] || 'px' : 'px';
      var sign = mv === 0 ? '' : mv < 0 ? "-=" + Math.abs(mv) + mu : "+=" + mv + mu;
      var start = "top " + startPct + "%" + sign;
      var removeHover = function () {
        if (hoverHandlerRef.current && ref.current) {
          ref.current.removeEventListener('mouseenter', hoverHandlerRef.current);
          hoverHandlerRef.current = null;
        }
      };
      var teardown = function () {
        if (tlRef.current) {
          tlRef.current.kill();
          tlRef.current = null;
        }
        if (wrappersRef.current.length) {
          wrappersRef.current.forEach(function (wrap) {
            var inner = wrap.firstElementChild;
            var orig = inner?.querySelector('[data-orig="1"]');
            if (orig && wrap.parentNode) wrap.parentNode.replaceChild(orig, wrap);
          });
          wrappersRef.current = [];
        }
        try {
          splitRef.current?.revert();
        } catch {
          /* noop */
        }
        splitRef.current = null;
        playingRef.current = false;
      };
      var build = function () {
        teardown();
        splitRef.current = new GSAPSplitText(el, {
          type: 'chars',
          charsClass: 'shuffle-char',
          wordsClass: 'shuffle-word',
          linesClass: 'shuffle-line',
          smartWrap: true,
          reduceWhiteSpace: false
        });
        var chars = splitRef.current.chars || [];
        wrappersRef.current = [];
        var rolls = Math.max(1, Math.floor(shuffleTimes));
        var rand = function (set) { return set.charAt(Math.floor(Math.random() * set.length)) || ''; };
        chars.forEach(function (ch) {
          var parent = ch.parentElement;
          if (!parent) return;
          var w = ch.getBoundingClientRect().width;
          if (!w) return;
          var wrap = document.createElement('span');
          Object.assign(wrap.style, {
            display: 'inline-block',
            overflow: 'hidden',
            width: w + 'px',
            verticalAlign: 'baseline'
          });
          var inner = document.createElement('span');
          Object.assign(inner.style, {
            display: 'inline-block',
            whiteSpace: 'nowrap',
            willChange: 'transform'
          });
          parent.insertBefore(wrap, ch);
          wrap.appendChild(inner);
          var firstOrig = ch.cloneNode(true);
          Object.assign(firstOrig.style, { display: 'inline-block', width: w + 'px', textAlign: 'center' });
          ch.setAttribute('data-orig', '1');
          Object.assign(ch.style, { display: 'inline-block', width: w + 'px', textAlign: 'center' });
          inner.appendChild(firstOrig);
          for (var k = 0; k < rolls; k++) {
            var c = ch.cloneNode(true);
            if (scrambleCharset) c.textContent = rand(scrambleCharset);
            Object.assign(c.style, { display: 'inline-block', width: w + 'px', textAlign: 'center' });
            inner.appendChild(c);
          }
          inner.appendChild(ch);
          var steps = rolls + 1;
          var startX = 0;
          var finalX = -steps * w;
          if (shuffleDirection === 'right') {
            var firstCopy = inner.firstElementChild;
            var real = inner.lastElementChild;
            if (real) inner.insertBefore(real, inner.firstChild);
            if (firstCopy) inner.appendChild(firstCopy);
            startX = -steps * w;
            finalX = 0;
          }
          gsap.set(inner, { x: startX, force3D: true });
          if (colorFrom) inner.style.color = colorFrom;
          inner.setAttribute('data-final-x', String(finalX));
          inner.setAttribute('data-start-x', String(startX));
          wrappersRef.current.push(wrap);
        });
      };
      var inners = function () { return wrappersRef.current.map(function (w) { return w.firstElementChild; }); };
      var randomizeScrambles = function () {
        if (!scrambleCharset) return;
        wrappersRef.current.forEach(function (w) {
          var strip = w.firstElementChild;
          if (!strip) return;
          var kids = Array.from(strip.children);
          for (var i = 1; i < kids.length - 1; i++) {
            kids[i].textContent = scrambleCharset.charAt(Math.floor(Math.random() * scrambleCharset.length));
          }
        });
      };
      var cleanupToStill = function () {
        wrappersRef.current.forEach(function (w) {
          var strip = w.firstElementChild;
          if (!strip) return;
          var real = strip.querySelector('[data-orig="1"]');
          if (!real) return;
          strip.replaceChildren(real);
          strip.style.transform = 'none';
          strip.style.willChange = 'auto';
        });
      };
      var play = function () {
        var strips = inners();
        if (!strips.length) return;
        playingRef.current = true;
        var tl = gsap.timeline({
          smoothChildTiming: true,
          repeat: loop ? -1 : 0,
          repeatDelay: loop ? loopDelay : 0,
          onRepeat: function () {
            if (scrambleCharset) randomizeScrambles();
            gsap.set(strips, { x: function (i, t) { return parseFloat(t.getAttribute('data-start-x') || '0'); } });
            onShuffleComplete?.();
          },
          onComplete: function () {
            playingRef.current = false;
            if (!loop) {
              cleanupToStill();
              if (colorTo) gsap.set(strips, { color: colorTo });
              onShuffleComplete?.();
              armHover();
            }
          }
        });
        var addTween = function (targets, at) {
          tl.to(
            targets,
            {
              x: function (i, t) { return parseFloat(t.getAttribute('data-final-x') || '0'); },
              duration: duration,
              ease: ease,
              force3D: true,
              stagger: animationMode === 'evenodd' ? stagger : 0
            },
            at
          );
          if (colorFrom && colorTo) {
            tl.to(targets, { color: colorTo, duration: duration, ease: ease }, at);
          }
        };
        if (animationMode === 'evenodd') {
          var odd = strips.filter(function (_, i) { return i % 2 === 1; });
          var even = strips.filter(function (_, i) { return i % 2 === 0; });
          var oddTotal = duration + Math.max(0, odd.length - 1) * stagger;
          var evenStart = odd.length ? oddTotal * 0.7 : 0;
          if (odd.length) addTween(odd, 0);
          if (even.length) addTween(even, evenStart);
        } else {
          strips.forEach(function (strip) {
            var d = Math.random() * maxDelay;
            tl.to(
              strip,
              {
                x: parseFloat(strip.getAttribute('data-final-x') || '0'),
                duration: duration,
                ease: ease,
                force3D: true
              },
              d
            );
            if (colorFrom && colorTo) tl.fromTo(strip, { color: colorFrom }, { color: colorTo, duration: duration, ease: ease }, d);
          });
        }
        tlRef.current = tl;
      };
      var armHover = function () {
        if (!triggerOnHover || !ref.current) return;
        removeHover();
        var handler = function () {
          if (playingRef.current) return;
          build();
          if (scrambleCharset) randomizeScrambles();
          play();
        };
        hoverHandlerRef.current = handler;
        ref.current.addEventListener('mouseenter', handler);
      };
      var create = function () {
        build();
        if (scrambleCharset) randomizeScrambles();
        play();
        armHover();
        ready = true;
        el.classList.add('is-ready');
      };
      var st = ScrollTrigger.create({
        trigger: el,
        start: start,
        once: triggerOnce,
        onEnter: create
      });
      var cleanup = function () {
        st.kill();
        removeHover();
        teardown();
        ready = false;
        if (el) el.classList.remove('is-ready');
      };
      var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
          if (mutation.type === 'childList' && !document.contains(el)) {
            cleanup();
          }
        });
      });
      observer.observe(document.body, { childList: true, subtree: true });
      return cleanup;
    }

    var commonStyle = { textAlign: textAlign, ...style };
    var classes = "shuffle-parent " + (ready ? 'is-ready' : '') + " " + className;
    var Tag = tag || 'p';
    var element = document.createElement(Tag);
    element.className = classes;
    Object.assign(element.style, commonStyle);
    element.textContent = text;
    ref.current = element;
    runAnimation();
    return element;
  };
})();

var PixelBlast = (function () {
  var THREE = window.THREE;
  var POSTPROCESSING = window.POSTPROCESSING;

  function createTouchTexture() {
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context not available');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const texture = new THREE.Texture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    const trail = [];
    let last = null;
    const maxAge = 64;
    let radius = 0.1 * size;
    const speed = 1 / maxAge;
    const clear = () => {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };
    const drawPoint = p => {
      const pos = { x: p.x * size, y: (1 - p.y) * size };
      let intensity = 1;
      const easeOutSine = t => Math.sin((t * Math.PI) / 2);
      const easeOutQuad = t => -t * (t - 2);
      if (p.age < maxAge * 0.3) intensity = easeOutSine(p.age / (maxAge * 0.3));
      else intensity = easeOutQuad(1 - (p.age - maxAge * 0.3) / (maxAge * 0.7)) || 0;
      intensity *= p.force;
      const color = `${((p.vx + 1) / 2) * 255}, ${((p.vy + 1) / 2) * 255}, ${intensity * 255}`;
      const offset = size * 5;
      ctx.shadowOffsetX = offset;
      ctx.shadowOffsetY = offset;
      ctx.shadowBlur = radius;
      ctx.shadowColor = `rgba(${color},${0.22 * intensity})`;
      ctx.beginPath();
      ctx.fillStyle = 'rgba(255,0,0,1)';
      ctx.arc(pos.x - offset, pos.y - offset, radius, 0, Math.PI * 2);
      ctx.fill();
    };
    const addTouch = norm => {
      let force = 0;
      let vx = 0;
      let vy = 0;
      if (last) {
        const dx = norm.x - last.x;
        const dy = norm.y - last.y;
        if (dx === 0 && dy === 0) return;
        const dd = dx * dx + dy * dy;
        const d = Math.sqrt(dd);
        vx = dx / (d || 1);
        vy = dy / (d || 1);
        force = Math.min(dd * 10000, 1);
      }
      last = { x: norm.x, y: norm.y };
      trail.push({ x: norm.x, y: norm.y, age: 0, force, vx, vy });
    };
    const update = () => {
      clear();
      for (let i = trail.length - 1; i >= 0; i--) {
        const point = trail[i];
        const f = point.force * speed * (1 - point.age / maxAge);
        point.x += point.vx * f;
        point.y += point.vy * f;
        point.age++;
        if (point.age > maxAge) trail.splice(i, 1);
      }
      for (let i = 0; i < trail.length; i++) drawPoint(trail[i]);
      texture.needsUpdate = true;
    };
    return {
      canvas,
      texture,
      addTouch,
      update,
      set radiusScale(v) {
        radius = 0.1 * size * v;
      },
      get radiusScale() {
        return radius / (0.1 * size);
      },
      size
    };
  }

  function createLiquidEffect(texture, opts) {
    const fragment = `
      uniform sampler2D uTexture;
      uniform float uStrength;
      uniform float uTime;
      uniform float uFreq;

      void mainUv(inout vec2 uv) {
        vec4 tex = texture2D(uTexture, uv);
        float vx = tex.r * 2.0 - 1.0;
        float vy = tex.g * 2.0 - 1.0;
        float intensity = tex.b;

        float wave = 0.5 + 0.5 * sin(uTime * uFreq + intensity * 6.2831853);

        float amt = uStrength * intensity * wave;

        uv += vec2(vx, vy) * amt;
      }
    `;
    return new POSTPROCESSING.Effect('LiquidEffect', fragment, {
      uniforms: new Map([
        ['uTexture', new THREE.Uniform(texture)],
        ['uStrength', new THREE.Uniform(opts?.strength ?? 0.025)],
        ['uTime', new THREE.Uniform(0)],
        ['uFreq', new THREE.Uniform(opts?.freq ?? 4.5)]
      ])
    });
  }

  const SHAPE_MAP = {
    square: 0,
    circle: 1,
    triangle: 2,
    diamond: 3
  };

  const VERTEX_SRC = `
    void main() {
      gl_Position = vec4(position, 1.0);
    }
  `;

  const FRAGMENT_SRC = `
    precision highp float;

    uniform vec3  uColor;
    uniform vec2  uResolution;
    uniform float uTime;
    uniform float uPixelSize;
    uniform float uScale;
    uniform float uDensity;
    uniform float uPixelJitter;
    uniform int   uEnableRipples;
    uniform float uRippleSpeed;
    uniform float uRippleThickness;
    uniform float uRippleIntensity;
    uniform float uEdgeFade;

    uniform int   uShapeType;
    const int SHAPE_SQUARE   = 0;
    const int SHAPE_CIRCLE   = 1;
    const int SHAPE_TRIANGLE = 2;
    const int SHAPE_DIAMOND  = 3;

    const int   MAX_CLICKS = 10;

    uniform vec2  uClickPos  [MAX_CLICKS];
    uniform float uClickTimes[MAX_CLICKS];

    out vec4 fragColor;

    float Bayer2(vec2 a) {
      a = floor(a);
      return fract(a.x / 2. + a.y * a.y * .75);
    }
    #define Bayer4(a) (Bayer2(.5*(a))*0.25 + Bayer2(a))
    #define Bayer8(a) (Bayer4(.5*(a))*0.25 + Bayer2(a))

    #define FBM_OCTAVES     5
    #define FBM_LACUNARITY  1.25
    #define FBM_GAIN        1.0

    float hash11(float n){ return fract(sin(n)*43758.5453); }

    float vnoise(vec3 p){
      vec3 ip = floor(p);
      vec3 fp = fract(p);
      float n000 = hash11(dot(ip + vec3(0.0,0.0,0.0), vec3(1.0,57.0,113.0)));
      float n100 = hash11(dot(ip + vec3(1.0,0.0,0.0), vec3(1.0,57.0,113.0)));
      float n010 = hash11(dot(ip + vec3(0.0,1.0,0.0), vec3(1.0,57.0,113.0)));
      float n110 = hash11(dot(ip + vec3(1.0,1.0,0.0), vec3(1.0,57.0,113.0)));
      float n001 = hash11(dot(ip + vec3(0.0,0.0,1.0), vec3(1.0,57.0,113.0)));
      float n101 = hash11(dot(ip + vec3(1.0,0.0,1.0), vec3(1.0,57.0,113.0)));
      float n011 = hash11(dot(ip + vec3(0.0,1.0,1.0), vec3(1.0,57.0,113.0)));
      float n111 = hash11(dot(ip + vec3(1.0,1.0,1.0), vec3(1.0,57.0,113.0)));
      vec3 w = fp*fp*fp*(fp*(fp*6.0-15.0)+10.0);
      float x00 = mix(n000, n100, w.x);
      float x10 = mix(n010, n110, w.x);
      float x01 = mix(n001, n101, w.x);
      float x11 = mix(n011, n111, w.x);
      float y0  = mix(x00, x10, w.y);
      float y1  = mix(x01, x11, w.y);
      return mix(y0, y1, w.z) * 2.0 - 1.0;
    }

    float fbm2(vec2 uv, float t){
      vec3 p = vec3(uv * uScale, t);
      float amp = 1.0;
      float freq = 1.0;
      float sum = 1.0;
      for (int i = 0; i < FBM_OCTAVES; ++i){
        sum  += amp * vnoise(p * freq);
        freq *= FBM_LACUNARITY;
        amp  *= FBM_GAIN;
      }
      return sum * 0.5 + 0.5;
    }

    float maskCircle(vec2 p, float cov){
      float r = sqrt(cov) * .25;
      float d = length(p - 0.5) - r;
      float aa = 0.5 * fwidth(d);
      return cov * (1.0 - smoothstep(-aa, aa, d * 2.0));
    }

    float maskTriangle(vec2 p, vec2 id, float cov){
      bool flip = mod(id.x + id.y, 2.0) > 0.5;
      if (flip) p.x = 1.0 - p.x;
      float r = sqrt(cov);
      float d  = p.y - r*(1.0 - p.x);
      float aa = fwidth(d);
      return cov * clamp(0.5 - d/aa, 0.0, 1.0);
    }

    float maskDiamond(vec2 p, float cov){
      float r = sqrt(cov) * 0.564;
      return step(abs(p.x - 0.49) + abs(p.y - 0.49), r);
    }

    void main(){
      float pixelSize = uPixelSize;
      vec2 fragCoord = gl_FragCoord.xy - uResolution * .5;
      float aspectRatio = uResolution.x / uResolution.y;

      vec2 pixelId = floor(fragCoord / pixelSize);
      vec2 pixelUV = fract(fragCoord / pixelSize);

      float cellPixelSize = 8.0 * pixelSize;
      vec2 cellId = floor(fragCoord / cellPixelSize);
      vec2 cellCoord = cellId * cellPixelSize;
      vec2 uv = cellCoord / uResolution * vec2(aspectRatio, 1.0);

      float base = fbm2(uv, uTime * 0.05);
      base = base * 0.5 - 0.65;

      float feed = base + (uDensity - 0.5) * 0.3;

      float speed     = uRippleSpeed;
      float thickness = uRippleThickness;
      const float dampT     = 1.0;
      const float dampR     = 10.0;

      if (uEnableRipples == 1) {
        for (int i = 0; i < MAX_CLICKS; ++i){
          vec2 pos = uClickPos[i];
          if (pos.x < 0.0) continue;
          float cellPixelSize = 8.0 * pixelSize;
          vec2 cuv = (((pos - uResolution * .5 - cellPixelSize * .5) / (uResolution))) * vec2(aspectRatio, 1.0);
          float t = max(uTime - uClickTimes[i], 0.0);
          float r = distance(uv, cuv);
          float waveR = speed * t;
          float ring  = exp(-pow((r - waveR) / thickness, 2.0));
          float atten = exp(-dampT * t) * exp(-dampR * r);
          feed = max(feed, ring * atten * uRippleIntensity);
        }
      }

      float bayer = Bayer8(fragCoord / uPixelSize) - 0.5;
      float bw = step(0.5, feed + bayer);

      float h = fract(sin(dot(floor(fragCoord / uPixelSize), vec2(127.1, 311.7))) * 43758.5453);
      float jitterScale = 1.0 + (h - 0.5) * uPixelJitter;
      float coverage = bw * jitterScale;
      float M;
      if      (uShapeType == SHAPE_CIRCLE)   M = maskCircle (pixelUV, coverage);
      else if (uShapeType == SHAPE_TRIANGLE) M = maskTriangle(pixelUV, pixelId, coverage);
      else if (uShapeType == SHAPE_DIAMOND)  M = maskDiamond(pixelUV, coverage);
      else                                   M = coverage;

      if (uEdgeFade > 0.0) {
        vec2 norm = gl_FragCoord.xy / uResolution;
        float edge = min(min(norm.x, norm.y), min(1.0 - norm.x, 1.0 - norm.y));
        float fade = smoothstep(0.0, uEdgeFade, edge);
        M *= fade;
      }

      vec3 color = uColor;
      fragColor = vec4(color, M);
    }
  `;

  const MAX_CLICKS = 10;

  return function PixelBlast({
    containerId = 'pixel-blast-background',
    variant = 'square',
    pixelSize = 3,
    color = '#B19EEF',
    antialias = true,
    patternScale = 2,
    patternDensity = 1,
    liquid = false,
    liquidStrength = 0.1,
    liquidRadius = 1,
    pixelSizeJitter = 0,
    enableRipples = true,
    rippleIntensityScale = 1,
    rippleThickness = 0.1,
    rippleSpeed = 0.3,
    liquidWobbleSpeed = 4.5,
    autoPauseOffscreen = true,
    speed = 0.5,
    transparent = true,
    edgeFade = 0.5,
    noiseAmount = 0
  }) {
    const container = document.getElementById(containerId);
    if (!container) return { cleanup: () => {} };

    let threeState = null;
    const visibilityState = { visible: true };
    let speedRef = speed;

    const needsReinitKeys = ['antialias', 'liquid', 'noiseAmount'];
    const cfg = { antialias, liquid, noiseAmount };
    let prevConfig = null;

    function init() {
      let mustReinit = !threeState;
      if (threeState && prevConfig) {
        for (const k of needsReinitKeys) {
          if (prevConfig[k] !== cfg[k]) {
            mustReinit = true;
            break;
          }
        }
      }

      if (mustReinit) {
        if (threeState) {
          threeState.resizeObserver?.disconnect();
          cancelAnimationFrame(threeState.raf);
          threeState.quad?.geometry.dispose();
          threeState.material.dispose();
          threeState.composer?.dispose();
          threeState.renderer.dispose();
          if (threeState.renderer.domElement.parentElement === container) {
            container.removeChild(threeState.renderer.domElement);
          }
          threeState = null;
        }

        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2', { antialias, alpha: true });
        if (!gl) return { cleanup: () => {} };

        const renderer = new THREE.WebGLRenderer({
          canvas,
          context: gl,
          antialias,
          alpha: true
        });
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        container.appendChild(renderer.domElement);

        const uniforms = {
          uResolution: { value: new THREE.Vector2(0, 0) },
          uTime: { value: 0 },
          uColor: { value: new THREE.Color(color) },
          uClickPos: {
            value: Array.from({ length: MAX_CLICKS }, () => new THREE.Vector2(-1, -1))
          },
          uClickTimes: { value: new Float32Array(MAX_CLICKS) },
          uShapeType: { value: SHAPE_MAP[variant] ?? 0 },
          uPixelSize: { value: pixelSize * renderer.getPixelRatio() },
          uScale: { value: patternScale },
          uDensity: { value: patternDensity },
          uPixelJitter: { value: pixelSizeJitter },
          uEnableRipples: { value: enableRipples ? 1 : 0 },
          uRippleSpeed: { value: rippleSpeed },
          uRippleThickness: { value: rippleThickness },
          uRippleIntensity: { value: rippleIntensityScale },
          uEdgeFade: { value: edgeFade }
        };

        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const material = new THREE.ShaderMaterial({
          vertexShader: VERTEX_SRC,
          fragmentShader: FRAGMENT_SRC,
          uniforms,
          transparent: true,
          glslVersion: THREE.GLSL3,
          depthTest: false,
          depthWrite: false
        });
        const quadGeom = new THREE.PlaneGeometry(2, 2);
        const quad = new THREE.Mesh(quadGeom, material);
        scene.add(quad);

        const clock = new THREE.Clock();

        const setSize = () => {
          const w = container.clientWidth || 1;
          const h = container.clientHeight || 1;
          renderer.setSize(w, h, false);
          uniforms.uResolution.value.set(renderer.domElement.width, renderer.domElement.height);
          if (threeState?.composer) {
            threeState.composer.setSize(renderer.domElement.width, renderer.domElement.height);
          }
          uniforms.uPixelSize.value = pixelSize * renderer.getPixelRatio();
        };
        setSize();

        const ro = new ResizeObserver(setSize);
        ro.observe(container);

        const randomFloat = () => {
          if (typeof window !== 'undefined' && window.crypto?.getRandomValues) {
            const u32 = new Uint32Array(1);
            window.crypto.getRandomValues(u32);
            return u32[0] / 0xffffffff;
          }
          return Math.random();
        };

        const timeOffset = randomFloat() * 1000;
        let composer;
        let touch;
        let liquidEffect;

        if (liquid) {
          touch = createTouchTexture();
          touch.radiusScale = liquidRadius;
          composer = new POSTPROCESSING.EffectComposer(renderer);
          const renderPass = new POSTPROCESSING.RenderPass(scene, camera);
          liquidEffect = createLiquidEffect(touch.texture, {
            strength: liquidStrength,
            freq: liquidWobbleSpeed
          });
          const effectPass = new POSTPROCESSING.EffectPass(camera, liquidEffect);
          effectPass.renderToScreen = true;
          composer.addPass(renderPass);
          composer.addPass(effectPass);
        }

        if (noiseAmount > 0) {
          if (!composer) {
            composer = new POSTPROCESSING.EffectComposer(renderer);
            composer.addPass(new POSTPROCESSING.RenderPass(scene, camera));
          }
          const noiseEffect = new POSTPROCESSING.Effect(
            'NoiseEffect',
            `uniform float uTime; uniform float uAmount; float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453);} void mainUv(inout vec2 uv){} void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){ float n=hash(floor(uv*vec2(1920.0,1080.0))+floor(uTime*60.0)); float g=(n-0.5)*uAmount; outputColor=inputColor+vec4(vec3(g),0.0);} `,
            {
              uniforms: new Map([
                ['uTime', new THREE.Uniform(0)],
                ['uAmount', new THREE.Uniform(noiseAmount)]
              ])
            }
          );
          const noisePass = new POSTPROCESSING.EffectPass(camera, noiseEffect);
          noisePass.renderToScreen = true;
          if (composer && composer.passes.length > 0) composer.passes.forEach(p => (p.renderToScreen = false));
          composer.addPass(noisePass);
        }

        if (composer) composer.setSize(renderer.domElement.width, renderer.domElement.height);

        const mapToPixels = e => {
          const rect = renderer.domElement.getBoundingClientRect();
          const scaleX = renderer.domElement.width / rect.width;
          const scaleY = renderer.domElement.height / rect.height;
          const fx = (e.clientX - rect.left) * scaleX;
          const fy = (rect.height - (e.clientY - rect.top)) * scaleY;
          return {
            fx,
            fy,
            w: renderer.domElement.width,
            h: renderer.domElement.height
          };
        };

        const onPointerDown = e => {
          const { fx, fy } = mapToPixels(e);
          const ix = threeState?.clickIx ?? 0;
          uniforms.uClickPos.value[ix].set(fx, fy);
          uniforms.uClickTimes.value[ix] = uniforms.uTime.value;
          if (threeState) threeState.clickIx = (ix + 1) % MAX_CLICKS;
        };

        const onPointerMove = e => {
          if (!touch) return;
          const { fx, fy, w, h } = mapToPixels(e);
          touch.addTouch({ x: fx / w, y: fy / h });
        };

        renderer.domElement.addEventListener('pointerdown', onPointerDown, { passive: true });
        renderer.domElement.addEventListener('pointermove', onPointerMove, { passive: true });

        let raf = 0;
        const animate = () => {
          if (autoPauseOffscreen && !visibilityState.visible) {
            raf = requestAnimationFrame(animate);
            return;
          }
          uniforms.uTime.value = timeOffset + clock.getElapsedTime() * speedRef;
          if (liquidEffect) liquidEffect.uniforms.get('uTime').value = uniforms.uTime.value;
          if (composer) {
            if (touch) touch.update();
            composer.passes.forEach(p => {
              const effs = p.effects;
              if (effs) {
                effs.forEach(eff => {
                  const u = eff.uniforms?.get('uTime');
                  if (u) u.value = uniforms.uTime.value;
                });
              }
            });
            composer.render();
          } else renderer.render(scene, camera);
          raf = requestAnimationFrame(animate);
        };
        raf = requestAnimationFrame(animate);

        threeState = {
          renderer,
          scene,
          camera,
          material,
          clock,
          clickIx: 0,
          uniforms,
          resizeObserver: ro,
          raf,
          quad,
          timeOffset,
          composer,
          touch,
          liquidEffect
        };
      } else {
        threeState.uniforms.uShapeType.value = SHAPE_MAP[variant] ?? 0;
        threeState.uniforms.uPixelSize.value = pixelSize * threeState.renderer.getPixelRatio();
        threeState.uniforms.uColor.value.set(color);
        threeState.uniforms.uScale.value = patternScale;
        threeState.uniforms.uDensity.value = patternDensity;
        threeState.uniforms.uPixelJitter.value = pixelSizeJitter;
        threeState.uniforms.uEnableRipples.value = enableRipples ? 1 : 0;
        threeState.uniforms.uRippleIntensity.value = rippleIntensityScale;
        threeState.uniforms.uRippleThickness.value = rippleThickness;
        threeState.uniforms.uRippleSpeed.value = rippleSpeed;
        threeState.uniforms.uEdgeFade.value = edgeFade;
        if (transparent) threeState.renderer.setClearAlpha(0);
        else threeState.renderer.setClearColor(0x000000, 1);
        if (threeState.liquidEffect) {
          const uStrength = threeState.liquidEffect.uniforms.get('uStrength');
          if (uStrength) uStrength.value = liquidStrength;
          const uFreq = threeState.liquidEffect.uniforms.get('uFreq');
          if (uFreq) uFreq.value = liquidWobbleSpeed;
        }
        if (threeState.touch) threeState.touch.radiusScale = liquidRadius;
      }

      prevConfig = cfg;

      return function cleanup() {
        if (!threeState) return;
        threeState.resizeObserver?.disconnect();
        cancelAnimationFrame(threeState.raf);
        threeState.quad?.geometry.dispose();
        threeState.material.dispose();
        threeState.composer?.dispose();
        threeState.renderer.dispose();
        if (threeState.renderer.domElement.parentElement === container) {
          container.removeChild(threeState.renderer.domElement);
        }
        threeState = null;
      };
    }

    const cleanup = init();
    return { element: container, cleanup };
  };
})();

(function () {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Shuffle, PixelBlast };
  } else {
    window.Shuffle = Shuffle;
    window.PixelBlast = PixelBlast;
  }

  // PixelBlast auto-init disabled: the app will not auto-render the background canvas.
  // To enable, call PixelBlast(...) from your app with a valid container element.
})();