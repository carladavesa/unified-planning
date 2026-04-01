;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
;; Reference paper: Scala, Enrico, Patrik Haslum, Sylvie Thiébaux, and Miquel Ramirez. 
;;                  "Subgoaling techniques for satisficing and optimal numeric planning." 
;;                  Journal of Artificial Intelligence Research 68 (2020): 691-752.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Block grouping domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; A number of blocks of different colours lie on a grid-like environment.
;;; The objective is to group the blocks by colour, i.e. to have all blocks
;;; of the same color in the same cell, which is at the same time
;;; different to the cell where blocks of other colors are:
;;;
;;; forall i, j color(i) = color(j) <=> loc(i) = loc(j)
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain mt-block-grouping)
    (:types block - object )
    (:functions
        (x ?b - block)  ;; The position of a block
        (y ?b - block)  ;;
        (maxx)
        (minx)
        (maxy)
        (miny)
    )

    ;; Move a block from its location to an adjacent location
    (:action moveblockup
     :parameters (?b - block)
     :precondition (and (<= (+ (y ?b)1) (maxy) ))
     :effect (and
        (increase (y ?b) 1)
    ))

    (:action moveblockdown
     :parameters (?b - block)
     :precondition (and (>= (- (y ?b) 1) (miny) ))
     :effect (and
        (decrease (y ?b) 1)
    ))

    (:action moveblockright
     :parameters (?b - block)
     :precondition (and (<= (+ (x ?b)1) (maxx) ))
     :effect (and
        (increase (x ?b) 1)
    ))

    (:action moveblockleft
     :parameters (?b - block)
     :precondition (and (>= (- (x ?b) 1) (minx) ))
     :effect (and
        (decrease (x ?b) 1)
    ))

)
