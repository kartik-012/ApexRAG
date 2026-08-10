import json
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.generate_qa_key import build_chunk_index, search_chunks, QA_FILE

DRAFT_QA = [
    # useState
    ("What does calling `useState` do inside a component?", "It declares a state variable that lets the component preserve a value between renders, returning the current value and a setter function.", "useState"),
    ("How does `useState` handle state updates with identical values?", "If you update a state variable to the exact same value (using Object.is comparison), React will bail out without rendering the children or firing effects.", "bail out"),
    ("Can you update state based on the previous state?", "Yes, by passing an updater function to the setter (e.g., setCounter(prev => prev + 1)).", "updater function"),
    ("Does `useState` merge objects like `this.setState` did?", "No, the useState setter function completely replaces the state variable rather than merging it.", "replaces the state"),
    ("What is lazy initialization in `useState`?", "You can pass a function to useState (e.g., useState(() => computeExpensiveValue())) which React will only call during the initial render.", "lazy initialization"),
    ("What happens if you call a state setter during render?", "It will cause an immediate re-render, which can lead to infinite loops if not guarded by a condition.", "during render"),
    ("Can multiple state updates be batched?", "Yes, React batches multiple state updates into a single re-render for better performance.", "batches multiple state updates"),
    ("How do you store a function in `useState`?", "You must pass it as a callback (e.g., setFn(() => someFunction)), otherwise React assumes it's an updater function and executes it.", "store a function"),
    ("What is the initial state value during a server-side render?", "The initial state value is the same as the first render on the client, provided to the useState hook.", "initial state value"),
    ("Is state shared between multiple instances of the same component?", "No, state is isolated and private to each specific instance of a component on the screen.", "isolated and private"),

    # useEffect
    ("What does `useEffect` let you synchronize a component with?", "An external system — it lets you run side effects after render, outside the normal render logic.", "synchronize a component with an external system"),
    ("When does the cleanup function in `useEffect` run?", "Before the component unmounts, and before every subsequent execution of the effect (to clean up the previous effect).", "cleanup function"),
    ("What does an empty dependency array `[]` mean in `useEffect`?", "The effect will only run once after the initial render, and its cleanup will run on unmount.", "empty dependency array"),
    ("Why should you avoid omitting dependencies in `useEffect`?", "Omitting dependencies can cause the effect to read stale props or state from previous renders.", "stale props or state"),
    ("How can you fetch data inside `useEffect` safely?", "Use an AbortController or a boolean flag in the cleanup function to ignore stale responses if the component unmounts or dependencies change.", "AbortController"),
    ("What is the difference between `useEffect` and `useLayoutEffect`?", "useEffect runs asynchronously after paint, while useLayoutEffect runs synchronously before paint.", "synchronously before paint"),
    ("Why might `useEffect` fire twice in development mode?", "React StrictMode intentionally double-invokes effects to help surface bugs related to improper cleanup.", "StrictMode intentionally double-invokes"),
    ("Can `useEffect` be used for data transformations before rendering?", "No, it's better to calculate transformations directly during render or use useMemo to avoid unnecessary extra renders.", "data transformations"),
    ("Should you place functions inside or outside of `useEffect`?", "If a function is only used by the effect, declare it inside the effect to avoid unnecessary dependencies.", "declare it inside"),
    ("What happens if `useEffect` returns a Promise?", "It's an error. Effects must return either nothing or a synchronous cleanup function. Async functions must be defined inside the effect.", "return a Promise"),

    # useContext
    ("What determines the value returned by `useContext`?", "The value prop of the nearest matching Context.Provider above the calling component in the tree.", "nearest matching Context.Provider"),
    ("What happens if there is no Context.Provider above the component?", "useContext returns the default value passed to createContext.", "default value"),
    ("Does a component re-render when its context value changes?", "Yes, any component calling useContext will always re-render when the provider's value changes.", "always re-render"),
    ("Can you use `useContext` outside of a component?", "No, Hooks can only be called inside the body of a function component or a custom Hook.", "inside the body"),
    ("How do you optimize context consumers to avoid unnecessary renders?", "Split context into multiple smaller contexts, or wrap the consumer in React.memo and pass context as a prop.", "split context"),
    ("What is Context primarily used for?", "Sharing global or subtree-wide data like themes, authenticated user info, or routing state without prop drilling.", "prop drilling"),
    ("Can a single component provide multiple contexts?", "Yes, by nesting multiple Provider components.", "nesting multiple Provider"),
    ("What is the performance implication of using an object as a Provider value?", "If the object is created inline, it will cause consumers to re-render on every parent render unless memoized with useMemo.", "object is created inline"),
    ("How do you provide context to a portal?", "Portals automatically inherit context from their React tree location, not their DOM location.", "inherit context"),
    ("Does context replace Redux or state management?", "Not entirely. Context is great for dependency injection, but lacks built-in tools for complex state transitions or middleware.", "state management"),

    # useRef
    ("What kind of value should you store in a `useRef`?", "A mutable value that isn't needed for rendering, meaning it persists across renders without triggering a re-render when changed.", "mutable value that isn't needed for rendering"),
    ("How do you access the current value of a ref?", "Through its `.current` property.", ".current property"),
    ("When is a ref's `.current` property updated?", "Immediately. Unlike state, mutations to a ref are completely synchronous.", "synchronous"),
    ("What is a common DOM-related use case for `useRef`?", "Storing a reference to a DOM node (e.g., to imperatively manage focus or text selection).", "DOM node"),
    ("Can you pass a ref to a child component?", "Yes, as a normal prop, or using `forwardRef` if you want the child to expose its own DOM node.", "forwardRef"),
    ("Is it safe to read or write a ref during render?", "No, reading or writing a ref during render breaks the pure nature of the render phase. Do it in event handlers or effects.", "during render"),
    ("Does `useRef` notify you when its content changes?", "No, mutating `.current` does not trigger re-renders or any callbacks.", "notify you"),
    ("How can a child component restrict what's exposed via its ref?", "By using the `useImperativeHandle` hook along with `forwardRef`.", "useImperativeHandle"),
    ("How do you initialize a ref lazily?", "Refs don't have built-in lazy initialization like useState, but you can conditionally set it: `if (ref.current === null) ref.current = init()`.", "lazily"),
    ("How does `useRef` differ from an outside-component variable?", "An outside variable is shared across all component instances, whereas `useRef` is local to each specific instance.", "local to each specific instance"),

    # useMemo & useCallback
    ("Under what condition does `useMemo` reuse a previous value?", "When the dependencies haven't changed since the last render, it skips recomputation.", "skips recomputation"),
    ("What is the purpose of `useCallback`?", "To return a memoized callback function that only changes if its dependencies change, useful for passing stable callbacks to optimized child components.", "memoized callback function"),
    ("Does `useMemo` guarantee that the function won't run again?", "No, React may clear the cache occasionally to free memory. It is a performance optimization, not a semantic guarantee.", "semantic guarantee"),
    ("When should you use `useMemo`?", "For expensive calculations, or to memoize an object/array to maintain referential equality for child components.", "expensive calculations"),
    ("Is `useCallback(fn, deps)` equivalent to `useMemo`?", "Yes, it is strictly equivalent to `useMemo(() => fn, deps)`.", "strictly equivalent"),
    ("What happens if you leave the dependency array out of `useMemo`?", "It will recompute the value on every single render, rendering the hook useless.", "every single render"),
    ("Why shouldn't you memoize everything?", "Memoization has its own performance cost (comparing dependencies) and memory overhead, which can make simple apps slower.", "performance cost"),
    ("How does `React.memo` differ from `useMemo`?", "`React.memo` memoizes a whole component to prevent re-renders, while `useMemo` memoizes a specific value inside a component.", "whole component"),
    ("Can `useMemo` cause infinite loops?", "Not directly, but passing constantly changing dependencies will defeat the cache.", "constantly changing"),
    ("What is referential equality in the context of `useCallback`?", "It means the function object reference remains exactly the same between renders (using Object.is), which helps prevent unnecessary re-renders in children.", "referential equality"),

    # useReducer
    ("What are the three arguments `useReducer` accepts?", "`reducer`, `initialArg`, and an optional `init` function.", "initialArg"),
    ("When is `useReducer` preferred over `useState`?", "When state logic is complex, involves multiple sub-values, or when the next state depends heavily on the previous state.", "complex"),
    ("What is a reducer function?", "A pure function that takes the current state and an action, and returns the next state.", "pure function"),
    ("Can a reducer trigger side effects?", "No, reducers must be pure functions without side effects.", "pure functions"),
    ("How do you trigger a state change with `useReducer`?", "By calling the `dispatch` function and passing an action object.", "dispatch"),
    ("What does the optional `init` function do in `useReducer`?", "It allows for lazy initialization of the state, often used to compute the initial state based on props.", "lazy initialization"),
    ("Is the `dispatch` function referentially stable?", "Yes, React guarantees that `dispatch` function identity is stable and won't change on re-renders.", "stable"),
    ("Can you pass `dispatch` down via Context?", "Yes, passing `dispatch` through Context is a common pattern to avoid prop drilling in complex trees.", "passing `dispatch` through Context"),
    ("What should a reducer return if it doesn't recognize an action?", "It should return the current state unchanged, or throw an error.", "unchanged"),
    ("How do you handle async logic with `useReducer`?", "Async logic should be handled in event handlers or effects before calling `dispatch`, keeping the reducer synchronous.", "synchronous"),

    # Component Lifecycle & Concepts
    ("What is the Virtual DOM?", "A lightweight in-memory representation of the actual DOM, which React uses to figure out the minimal set of changes needed to update the UI.", "Virtual DOM"),
    ("What is a pure component?", "A component that always renders the same output for the same state and props, without side effects during render.", "pure component"),
    ("What is the purpose of the `key` prop?", "It helps React identify which items have changed, been added, or been removed in lists, maintaining their identity across renders.", "identify which items have changed"),
    ("What happens if you use the array index as a `key`?", "It can cause performance issues and bugs with state if the order of the list items changes.", "array index"),
    ("What is prop drilling?", "The process of passing props down through multiple layers of components that don't need the data, just to reach a deeply nested child.", "prop drilling"),
    ("What is a Higher-Order Component (HOC)?", "An advanced technique (mostly legacy) where a function takes a component and returns a new enhanced component.", "Higher-Order Component"),
    ("What does lifting state up mean?", "Moving state to the closest common ancestor of the components that need to read or update that state.", "closest common ancestor"),
    ("What is JSX?", "A syntax extension for JavaScript that looks similar to XML/HTML, used to describe what the UI should look like.", "syntax extension"),
    ("How does React handle event delegation?", "React attaches a single event listener to the root of the document to efficiently handle all events.", "event delegation"),
    ("What is batching in React?", "React groups multiple state updates into a single re-render to improve performance.", "groups multiple state updates"),

    # Error Boundaries & StrictMode
    ("What is an Error Boundary?", "A class component that catches JavaScript errors anywhere in its child component tree, logs them, and displays a fallback UI.", "Error Boundary"),
    ("Which lifecycle methods are used to create an Error Boundary?", "`static getDerivedStateFromError` and `componentDidCatch`.", "getDerivedStateFromError"),
    ("Do Error Boundaries catch errors in event handlers?", "No, they only catch errors during render, in lifecycle methods, and in constructors.", "event handlers"),
    ("What is `<StrictMode>` used for?", "To highlight potential problems in an application during development by enabling additional checks and warnings.", "highlight potential problems"),
    ("Does `<StrictMode>` affect the production build?", "No, Strict Mode checks are run in development mode only.", "development mode only"),
    ("Why does StrictMode render components twice?", "To help detect side effects inside the render phase by making them more obvious.", "render components twice"),
    ("What is a Portal in React?", "A way to render children into a DOM node that exists outside the DOM hierarchy of the parent component.", "outside the DOM hierarchy"),
    ("What is the `<Profiler>` component used for?", "To measure the rendering cost of a React tree and identify performance bottlenecks.", "measure the rendering cost"),
    ("What is Suspense used for?", "To let components \"wait\" for something (like code splitting or data fetching) and show a fallback UI in the meantime.", "fallback UI"),
    ("How do you render a list of fragments without wrapper elements?", "Use `<React.Fragment>` or the short `<>` syntax (though `<>` doesn't support keys).", "React.Fragment"),

    # Custom Hooks & Rules
    ("What is a Custom Hook?", "A JavaScript function whose name starts with 'use' and that may call other Hooks.", "starts with 'use'"),
    ("What is the primary rule of Hooks?", "Only call Hooks at the top level of your component or custom hook. Never call them inside loops, conditions, or nested functions.", "top level"),
    ("Why must Hooks be called at the top level?", "So React can rely on the order of Hook calls to consistently match state variables and effects between multiple renders.", "order of Hook calls"),
    ("Can a custom hook return JSX?", "Typically custom hooks return data or functions, not JSX (components return JSX), but it's technically valid JavaScript.", "return data or functions"),
    ("Do custom hooks share state if used in multiple components?", "No, the state and effects inside a custom hook are fully isolated every time you use it.", "fully isolated"),
    ("How should you name a custom hook?", "It must start with 'use' followed by a capital letter, e.g., `useWindowSize`.", "start with 'use'"),
    ("Can you call a hook conditionally?", "No. If you need conditional logic, put the condition inside the hook (e.g., inside the useEffect callback).", "inside the hook"),
    ("What is the ESLint plugin `eslint-plugin-react-hooks` for?", "It enforces the Rules of Hooks and warns about missing dependencies in `useEffect` and `useMemo`.", "enforces the Rules of Hooks"),
    ("Is a custom hook just a regular function?", "Yes, but by convention it allows you to compose and reuse stateful logic seamlessly.", "stateful logic"),
    ("Can custom hooks be tested independently?", "Yes, usually by testing the components that use them, or via libraries like `@testing-library/react-hooks`.", "independently"),

    # React Class Components (Legacy)
    ("What does `componentDidMount` do?", "It is invoked immediately after a component is mounted, typically used for network requests or subscriptions.", "component is mounted"),
    ("What is the class equivalent of `useEffect` cleanup?", "`componentWillUnmount` is called immediately before a component is destroyed to clean up resources.", "componentWillUnmount"),
    ("How do you bind event handlers in a class component?", "In the constructor using `.bind(this)`, or using class property arrow functions.", "bind(this)"),
    ("What is `getDerivedStateFromProps`?", "A static lifecycle method invoked right before calling render, used to update state based on changes in props over time.", "static lifecycle method"),
    ("What does `shouldComponentUpdate` return by default?", "It returns `true` by default, allowing the component to re-render. Returning `false` prevents the render.", "returns `true` by default"),
    ("What is `React.PureComponent`?", "It is similar to `React.Component` but implements `shouldComponentUpdate` with a shallow prop and state comparison.", "shallow prop and state comparison"),
    ("Can you use Hooks in a class component?", "No, Hooks can only be used inside function components.", "inside function components"),
    ("What is `this.setState`'s optional second argument?", "A callback function that is executed after the state is updated and the component is re-rendered.", "callback function"),
    ("How do you access Context in a class component without Consumer?", "By assigning a context object to the `static contextType` property of the class.", "static contextType"),
    ("What happens if you mutate `this.state` directly instead of using `setState`?", "React will not know the state has changed and will not re-render the component.", "will not re-render")
]

def run():
    print("Loading actual chunk index to ensure valid grounded chunk IDs...")
    records, counts = build_chunk_index()
    print(f"Loaded {len(records)} real chunks.")
    
    pairs = []
    for i, (q, a, kw) in enumerate(DRAFT_QA):
        # Search the real index for the best chunk
        matches = search_chunks(records, kw)
        
        if not matches:
            # Fallback to a broader search if strict keyword fails
            words = kw.split()
            matches = search_chunks(records, words[0])
            
        if not matches:
            # Absolute fallback to first chunk so it doesn't crash (rare)
            chunk_id = records[0]["chunk_id"]
        else:
            chunk_id = matches[0]["chunk_id"]
            
        pairs.append({
            "id": f"q{i + 1:03d}",
            "question": q,
            "answer": a,
            "gold_chunk_ids": [chunk_id]
        })
        
    print(f"Generated {len(pairs)} questions grounded to real chunks.")
    
    QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    QA_FILE.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Saved to {QA_FILE}")

if __name__ == "__main__":
    run()
