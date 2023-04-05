    ! function (t) {
        var e = function (t, e, a) {
            var s;
            return function () {
                var i = this,
                    l = arguments,
                    n = function () { s = null, a || t.apply(i, l) },
                    o = a && !s;
                clearTimeout(s), s = setTimeout(n, e), o && t.apply(i, l)
            }
        };
        t(window)
            .on("elementor/frontend/init", function () {
                var t, a = elementorModules.frontend.handlers.Base;
                t = a.extend({
                    bindEvents: function () { this.run() },
                    getDefaultSettings: function () {
                        return {
                            direction: "alternate",
                            easing: "easeInOutSine",
                            loop: !0,
                            targets: this.findElement(".elementor-widget-container")
                                .get(0)
                        }
                    },
                    onElementChange: e(function (t) {-1 !== t.indexOf("ha_floating") && (this.anime && this.anime.restart(), this.run()) }, 400),
                    getFxVal: function (t) { return this.getElementSettings("ha_floating_fx_" + t) },
                    run: function () {
                        var t = this.getDefaultSettings();
                        this.getFxVal("translate_toggle") && ((this.getFxVal("translate_x.size") || this.getFxVal("translate_x.sizes.to")) && (t.translateX = { value: [this.getFxVal("translate_x.sizes.from") || 0, this.getFxVal("translate_x.size") || this.getFxVal("translate_x.sizes.to")], duration: this.getFxVal("translate_duration.size"), delay: this.getFxVal("translate_delay.size") || 0 }), (this.getFxVal("translate_y.size") || this.getFxVal("translate_y.sizes.to")) && (t.translateY = { value: [this.getFxVal("translate_y.sizes.from") || 0, this.getFxVal("translate_y.size") || this.getFxVal("translate_y.sizes.to")], duration: this.getFxVal("translate_duration.size"), delay: this.getFxVal("translate_delay.size") || 0 })), this.getFxVal("rotate_toggle") && ((this.getFxVal("rotate_x.size") || this.getFxVal("rotate_x.sizes.to")) && (t.rotateX = { value: [this.getFxVal("rotate_x.sizes.from") || 0, this.getFxVal("rotate_x.size") || this.getFxVal("rotate_x.sizes.to")], duration: this.getFxVal("rotate_duration.size"), delay: this.getFxVal("rotate_delay.size") || 0 }), (this.getFxVal("rotate_y.size") || this.getFxVal("rotate_y.sizes.to")) && (t.rotateY = { value: [this.getFxVal("rotate_y.sizes.from") || 0, this.getFxVal("rotate_y.size") || this.getFxVal("rotate_y.sizes.to")], duration: this.getFxVal("rotate_duration.size"), delay: this.getFxVal("rotate_delay.size") || 0 }), (this.getFxVal("rotate_z.size") || this.getFxVal("rotate_z.sizes.to")) && (t.rotateZ = { value: [this.getFxVal("rotate_z.sizes.from") || 0, this.getFxVal("rotate_z.size") || this.getFxVal("rotate_z.sizes.to")], duration: this.getFxVal("rotate_duration.size"), delay: this.getFxVal("rotate_delay.size") || 0 })), this.getFxVal("scale_toggle") && ((this.getFxVal("scale_x.size") || this.getFxVal("scale_x.sizes.to")) && (t.scaleX = { value: [this.getFxVal("scale_x.sizes.from") || 0, this.getFxVal("scale_x.size") || this.getFxVal("scale_x.sizes.to")], duration: this.getFxVal("scale_duration.size"), delay: this.getFxVal("scale_delay.size") || 0 }), (this.getFxVal("scale_y.size") || this.getFxVal("scale_y.sizes.to")) && (t.scaleY = { value: [this.getFxVal("scale_y.sizes.from") || 0, this.getFxVal("scale_y.size") || this.getFxVal("scale_y.sizes.to")], duration: this.getFxVal("scale_duration.size"), delay: this.getFxVal("scale_delay.size") || 0 })), (this.getFxVal("translate_toggle") || this.getFxVal("rotate_toggle") || this.getFxVal("scale_toggle")) && (this.findElement(".elementor-widget-container")
                            .css("will-change", "transform"), this.anime = window.anime && window.anime(t))
                    }
                }), elementorFrontend.hooks.addAction("frontend/element_ready/widget", function (e) { elementorFrontend.elementsHandler.addHandler(t, { $element: e }) })
            })
    }(jQuery); 