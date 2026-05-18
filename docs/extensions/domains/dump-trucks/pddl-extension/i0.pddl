(define (problem dump-trucks_i0)
    (:domain dump-trucks)

    (:objects
        l1 l2 - location
        t1 t2 - truck
        p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - package
    )

    (:init
        (truck_at t1 l1)
        (truck_at t2 l2)

        (= (package_at l1) (set.mk (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9)))

        (= (connects l1) (set.mk (l2)))
        (= (connects l2) (set.mk (l1)))

        (= (package_at l2) (set.mk ()))
        (= (package_in t1) (set.mk ()))
        (= (package_in t2) (set.mk ()))
    )

    (:goal
        (and
            (> (cardinality (union (package_in t1) (package_in t2))) 5)
            (< (cardinality (package_in t1)) (cardinality (package_in t2)))
        )
    )
)
